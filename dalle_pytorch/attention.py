from inspect import isfunction
from math import sqrt

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# classes

class Attention(nn.Module):
    def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0., noncausal_attn_len = 0):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.causal = causal
        self.noncausal_attn_len = noncausal_attn_len

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()

            if self.noncausal_attn_len > 1:
                ind = slice(0, self.noncausal_attn_len)
                mask[ind, ind] = False

            dots.masked_fill_(mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class SparseConvCausalAttention(nn.Module):
    def __init__(self, dim, seq_len, image_size = 32, kernel_size = 5, heads = 8, dim_head = 64, dropout = 0., **kwargs):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel size must be odd'

        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.image_size = image_size
        self.kernel_size = kernel_size

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h, img_size, kernel_size, device = *x.shape, self.heads, self.image_size, self.kernel_size, x.device
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        q *= self.scale

        img_seq_len = img_size ** 2
        text_len = n - img_seq_len
        ((q_text, q_img), (k_text, k_img), (v_text, v_img)) = map(lambda t: (t[:, img_seq_len:], t[:, -img_seq_len:]), (q, k, v))

        # text attention

        dots_text = einsum('b i d, b j d -> b i j', q_text, k_text)
        mask_value = max_neg_value(dots_text)

        i, j = dots_text.shape[-2:]
        mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
        dots_text.masked_fill(mask, mask_value)

        attn_text = dots_text.softmax(dim = -1)
        out_text = einsum('b i j, b j d -> b i d', attn_text, v_text)

        # image attention

        k_img, v_img = map(lambda t: rearrange(t, 'b (h w) c -> b c h w', h = img_size), (k_img, v_img))
        k_img, v_img = map(lambda t: F.unfold(t, kernel_size, padding = (kernel_size // 2)), (k_img, v_img))
        k_img, v_img = map(lambda t: rearrange(t, 'b (j d) i -> b i j d', j = kernel_size ** 2), (k_img, v_img))

        k_text, v_text = map(lambda t: repeat(t, 'b j d -> b i j d', i = img_seq_len), (k_text, v_text))

        # let image attend to all of text

        k_img = torch.cat((k_text, k_img), dim = 2)
        v_img = torch.cat((v_text, v_img), dim = 2)

        dots_image = einsum('b i d, b i j d -> b i j', q_img, k_img)

        # calculate causal attention for local convolution

        i, j = dots_image.shape[-2:]
        img_seq = torch.arange(img_seq_len, device = device)
        k_img_indices = rearrange(img_seq.float(), '(h w) -> () () h w', h = img_size)
        k_img_indices = F.pad(k_img_indices, (kernel_size // 2,) * 4, value = img_seq_len)
        k_img_indices = F.unfold(k_img_indices, kernel_size)
        k_img_indices = rearrange(k_img_indices, 'b j i -> b i j')

        # mask image attention

        mask = rearrange(img_seq, 'i -> () i ()') <= k_img_indices

        # image can attend to all of text

        mask = F.pad(mask, (text_len, 0), value = True)
        dots_image.masked_fill_(~mask, mask_value)

        attn_image = dots_image.softmax(dim = -1)
        out_image = einsum('b i j, b i j d -> b i d', attn_image, v_img)

        # combine attended values for both text and image

        out = torch.cat((out_text, out_image), dim = 1)

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out =  self.to_out(out)
        return out

class SparseAttention(Attention):
    def __init__(
        self,
        *args,
        block_size = 16,
        num_random_blocks = None,
        sparse_attn_global_indices = [],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        from deepspeed.ops.sparse_attention import SparseSelfAttention, VariableSparsityConfig
        self.block_size = block_size

        num_random_blocks = default(num_random_blocks, self.seq_len // block_size // 4)
        global_blocks = uniq(map(lambda t: t // block_size, sparse_attn_global_indices))

        self.attn_fn = SparseSelfAttention(
            sparsity_config = VariableSparsityConfig(
                num_heads = self.heads,
                block = self.block_size,
                num_random_blocks = num_random_blocks,
                global_block_indices = global_blocks,
                attention = 'unidirectional' if self.causal else 'bidirectional'
            ),
            max_seq_length = self.seq_len,
            attn_mask_mode = 'add'
        )

    def forward(self, x, mask = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        remainder = n % self.block_size
        mask = default(mask, lambda: torch.ones(b, n, device = device).bool())

        if remainder > 0:
            padding = self.block_size - remainder
            x = F.pad(x, (0, 0, 0, padding), value = 0)
            mask = F.pad(mask, (0, padding), value = False)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        key_pad_mask = None
        if exists(mask):
            key_pad_mask = ~mask

        attn_mask = None
        if self.causal:
            i, j = q.shape[-2], k.shape[-2]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            attn_mask = torch.zeros(i, j, device = device).to(q)
            mask_value = max_neg_value(q) / 2
            attn_mask.masked_fill_(mask, mask_value)

            if self.noncausal_attn_len:
                ind = slice(0, self.noncausal_attn_len)
                attn_mask[ind, ind] = 0.

        out = self.attn_fn(q, k, v, attn_mask = attn_mask, key_padding_mask = key_pad_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out[:, :n]
