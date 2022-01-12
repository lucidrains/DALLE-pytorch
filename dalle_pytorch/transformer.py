from collections import deque
from collections.abc import Iterable
from functools import partial
from itertools import islice, cycle

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from dalle_pytorch.reversible import ReversibleSequence, SequentialSequence
from dalle_pytorch.attention import Attention, SparseAttention, SparseConvCausalAttention, SparseAxialCausalAttention

from rotary_embedding_torch import RotaryEmbedding, broadcat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, depth = 1):
    return val if isinstance(val, Iterable) else (val,) * depth

# classes

class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim = self.dim, keepdim = True).detach()
        return x / maxes

class NonCached(nn.Module):
    """
    A wrapper for layers that don't support the inference cache themselves.
    Reconstructs the full sequence before the layer and
    cuts the suffix of the outputs after the layer.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *, cache = None, cache_key = None, **kwargs):
        n = x.shape[-2]
        if exists(cache):
            if cache_key in cache:
                x = torch.cat([cache[cache_key], x], dim=-2)
            cache[cache_key] = x

        out = self.fn(x, **kwargs)

        return out[:, -n:]

class CachedAs(nn.Module):
    """
    A wrapper that defines a key for the inference cache.
    """

    def __init__(self, cache_key, fn):
        super().__init__()
        self.cache_key = cache_key
        self.fn = fn

    def forward(self, x, *, cache=None, **kwargs):
        return self.fn(x, cache=cache, cache_key=self.cache_key, **kwargs)

# https://arxiv.org/abs/2103.17239
class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

# layer norm

class PreNorm(nn.Module):
    def __init__(self, dim, fn, sandwich = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)

# feed forward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0., mult = 4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, cache=None, cache_key=None):
        return self.net(x)

# token shift classes

class PreShiftToken(nn.Module):
    def __init__(self, fn, image_size, seq_len):
        super().__init__()
        self.fn = fn
        self.image_size = image_size
        self.seq_len = seq_len
        self.img_seq_len = image_size ** 2
        self.text_len = seq_len - self.img_seq_len + 1

    def forward(self, x, cache=None, cache_key=None, **kwargs):
        seq_len, image_size, text_len = self.seq_len, self.image_size, self.text_len

        if exists(cache) and cache_key in cache:
            offset = cache['offset']
            assert offset >= text_len, "cached inference for text is not supported"
            q = cache[cache_key]
            assert isinstance(q, deque) and len(q) == image_size

            x_top, x_left, *x_pass = x[:, -1].chunk(4, dim=-1)

            q.append((x_top, x_left))
            x_top = q.popleft()[0]
            x_left = q[-2][1]
            if (offset - text_len) % image_size == 0:
                x_left = torch.zeros_like(x_left)

            x = torch.cat((x_top, x_left, *x_pass), dim=-1)
            return self.fn(x[:, None], cache=cache, **kwargs)

        n = x.shape[1]
        padding = seq_len - n + 1

        # if sequence is shorter than the text length, no image tokens to shift

        if n < text_len:
            return self.fn(x, **kwargs)

        # get text and image tokens

        x_text, x_img = x[:, :text_len], x[:, text_len:]
        x_img = F.pad(x_img, (0, 0, 0, padding))
        x_img = rearrange(x_img, 'b (h w) d -> b h w d', h = image_size)

        # shift 1 from the left for text tokens

        x_text_shift, x_text_pass = x_text.chunk(2, dim = -1)
        x_text_shift = F.pad(x_text_shift, (0, 0, 1, -1))
        x_text = torch.cat((x_text_shift, x_text_pass), dim = -1)

        # shift from top, left for image tokens

        x_img_shift_top, x_img_shift_left, *x_img_pass = x_img.chunk(4, dim = -1)
        x_img_shift_left = F.pad(x_img_shift_left, (0, 0, 1, -1))
        x_img_shift_top = F.pad(x_img_shift_top, (0, 0, 0, 0, 1, -1))
        x_img = torch.cat((x_img_shift_top, x_img_shift_left, *x_img_pass), dim = -1)

        # merge text and image sequence back together

        x_img = rearrange(x_img, 'b h w d -> b (h w) d')
        x_img = x_img[:, :-padding]
        x = torch.cat((x_text, x_img), dim = 1)

        if exists(cache):
            dummy_top, dummy_left, *_ = x[:, -1].chunk(4, dim=-1)
            dummy_top, dummy_left = torch.zeros_like(dummy_top), torch.zeros_like(dummy_left)

            q = deque()
            x_img = x_img[:, -image_size:]
            for _ in range(image_size - x_img.shape[1]):
                q.append((dummy_top, dummy_left))
            for i in range(x_img.shape[1]):
                q.append(x_img[:, i].chunk(4, dim=-1)[:2])
            cache[cache_key] = q

        return self.fn(x, cache=cache, **kwargs)

# main transformer class

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        reversible = False,
        causal = True,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_types = None,
        image_fmap_size = None,
        sparse_attn = False,
        stable = False,
        sandwich_norm = False,
        shift_tokens = False,
        rotary_emb = True,
        shared_attn_ids = None,
        shared_ff_ids = None,
        optimize_for_inference = False,  # use cache-friendly masked attention instead of sparse one
    ):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)

        self.seq_len = seq_len
        self.image_fmap_size = image_fmap_size

        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        shared_attn_ids = cycle(default(shared_attn_ids, range(depth)))
        shared_ff_ids = cycle(default(shared_ff_ids, range(depth)))
        shared_attn_layers = {}
        shared_ff_layers = {}

        for (ind, sparse_attn, attn_type, attn_id, ff_id) in \
                zip(range(depth), sparse_layer, attn_type_layer, shared_attn_ids, shared_ff_ids):
            if attn_type == 'full':
                attn_class = partial(Attention, stable = stable)
            elif attn_type == 'sparse':
                attn_class = SparseAttention
            elif attn_type == 'axial_row':
                if optimize_for_inference:
                    attn_class = partial(Attention, stable = stable, static_mask = self._get_attention_mask(attn_type))
                else:
                    attn_class = partial(SparseAxialCausalAttention, seq_len = seq_len, axis = 0, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'axial_col':
                if optimize_for_inference:
                    attn_class = partial(Attention, stable = stable, static_mask = self._get_attention_mask(attn_type))
                else:
                    attn_class = partial(SparseAxialCausalAttention, seq_len = seq_len, axis = 1, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'conv_like':
                attn_class = partial(SparseConvCausalAttention, seq_len = seq_len, image_size = image_fmap_size, stable = stable)
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')

            attn, reused_attn_type = shared_attn_layers.get(attn_id, (None, None))
            if not exists(attn):
                attn = attn_class(dim, causal = causal, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)
                shared_attn_layers[attn_id] = (attn, attn_type)
            elif attn_type != reused_attn_type:
                raise ValueError('attn_types do not match shared_attn_ids '
                                 f'(ind = {ind}, attn_type = "{attn_type}", reused_attn_type = "{reused_attn_type}")')

            ff = shared_ff_layers.get(ff_id)
            if not exists(ff):
                ff = FeedForward(dim, mult = ff_mult, dropout = ff_dropout)
                shared_ff_layers[ff_id] = ff

            if isinstance(attn, Attention):
                attn = CachedAs(f'attn_{ind}', attn)
            else:
                # at the moment, other attention classes don't support cache
                attn = NonCached(attn)

            if shift_tokens:
                attn = CachedAs(f'preshift_attn_{ind}', PreShiftToken(attn, image_size = image_fmap_size, seq_len = seq_len))
                ff = CachedAs(f'preshift_ff_{ind}', PreShiftToken(ff, image_size = image_fmap_size, seq_len = seq_len))

            layers.append(nn.ModuleList([
                LayerScale(dim, ind + 1, PreNorm(dim, attn, sandwich = sandwich_norm)),
                LayerScale(dim, ind + 1, PreNorm(dim, ff, sandwich = sandwich_norm))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        route_all = ((True, True),) * depth
        attn_route_map = {'mask': route_attn, 'rotary_pos_emb': route_attn,
                          'cache': route_all}

        self.layers = execute_type(layers, args_route = attn_route_map)

        # generate positional embeddings for rotary

        pos_emb = None
        if rotary_emb:
            rot_dim = dim_head // 3
            img_seq_len = (image_fmap_size ** 2)
            text_len = seq_len - img_seq_len + 1

            text_pos_emb = RotaryEmbedding(dim = rot_dim)
            img_axial_pos_emb = RotaryEmbedding(dim = rot_dim, freqs_for = 'pixel')

            text_freqs = text_pos_emb(torch.arange(text_len))
            img_to_text_freqs = text_pos_emb(torch.full((img_seq_len,), 8192)) # image is given a position far away from text
            text_freqs = torch.cat((text_freqs, img_to_text_freqs), dim = 0)

            img_freqs_axial = img_axial_pos_emb(torch.linspace(-1, 1, steps = image_fmap_size))
            img_freqs = broadcat((rearrange(img_freqs_axial, 'i d -> i () d'), rearrange(img_freqs_axial, 'j d -> () j d')), dim = -1)
            img_freqs = rearrange(img_freqs, 'h w d -> (h w) d')

            text_axial_freqs = img_axial_pos_emb(torch.full((text_len,), -10.))  # text is given a position of -10 apart from the image axial positions, which is from range [-1, 1]
            text_axial_freqs = torch.cat((text_axial_freqs, text_axial_freqs), dim = -1)
            img_freqs = torch.cat((text_axial_freqs, img_freqs), dim = 0)

            pos_emb = torch.cat((text_freqs, img_freqs), dim = -1)
            pos_emb = rearrange(pos_emb, 'n d -> () n d')

        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x, **kwargs):
        return self.layers(x, rotary_pos_emb = self.pos_emb, **kwargs)

    def _get_attention_mask(self, attn_type):
        img_seq_len = self.image_fmap_size ** 2
        text_len = self.seq_len + 1 - img_seq_len

        static_mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)
        static_mask[:, :text_len] = True
        if attn_type == 'axial_row':
            for row in range(self.image_fmap_size):
                begin = text_len + row * self.image_fmap_size
                end = text_len + (row + 1) * self.image_fmap_size
                static_mask[begin:end, begin:end] = True
        elif attn_type == 'axial_col':
            for col in range(self.image_fmap_size):
                begin = text_len + col
                static_mask[begin::self.image_fmap_size, begin::self.image_fmap_size] = True
        else:
            raise ValueError(f'attention type "{attn_type}" can\'t be simulated with a static mask')
        return static_mask
