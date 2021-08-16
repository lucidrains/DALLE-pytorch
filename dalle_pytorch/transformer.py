from functools import partial
from itertools import islice, cycle

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from dalle_pytorch.reversible import ReversibleSequence, SequentialSequence
from dalle_pytorch.attention import Attention, SparseAttention, SparseConvCausalAttention, SparseAxialCausalAttention

from g_mlp_pytorch import gMLPBlock

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, depth = 1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim = self.dim, keepdim = True)
        return x / maxes

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
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

    def forward(self, x):
        return self.net(x)

# token shift classes

class PreShiftToken(nn.Module):
    def __init__(self, fn, image_size, seq_len):
        super().__init__()
        self.fn = fn
        self.image_size = image_size
        self.seq_len = seq_len

    def forward(self, x, **kwargs):
        n = x.shape[1]
        seq_len, image_size = self.seq_len, self.image_size
        img_seq_len = image_size ** 2
        text_len = seq_len - img_seq_len + 1
        padding = seq_len - n + 1

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
        x = torch.cat((x_text, x_img[:, :-padding]), dim = 1)
        return self.fn(x, **kwargs)

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
        shift_tokens = True
    ):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)

        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        for ind, sparse_attn, attn_type in zip(range(depth), sparse_layer, attn_type_layer):
            if attn_type == 'full':
                attn_class = partial(Attention, stable = stable)
            elif attn_type == 'sparse':
                attn_class = SparseAttention
            elif attn_type == 'axial_row':
                attn_class = partial(SparseAxialCausalAttention, seq_len = seq_len, axis = 0, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'axial_col':
                attn_class = partial(SparseAxialCausalAttention, seq_len = seq_len, axis = 1, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'conv_like':
                attn_class = partial(SparseConvCausalAttention, seq_len = seq_len, image_size = image_fmap_size, stable = stable)
            elif attn_type == 'mlp':
                attn_class = partial(gMLPBlock, seq_len = seq_len)
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')

            if attn_type != 'mlp':
                attn = attn_class(dim, causal = causal, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            else:
                attn = attn_class(dim = dim, causal = causal, dim_ff = dim * 4)

            ff = FeedForward(dim, mult = ff_mult, dropout = ff_dropout)

            if shift_tokens:
                attn, ff = map(lambda t: PreShiftToken(t, image_size = image_fmap_size, seq_len = seq_len), (attn, ff))

            layers.append(nn.ModuleList([
                LayerScale(dim, ind + 1, PreNorm(dim, attn)),
                LayerScale(dim, ind + 1, PreNorm(dim, ff))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}

        self.layers = execute_type(layers, args_route = attn_route_map)

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)
