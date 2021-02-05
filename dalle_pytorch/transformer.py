from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from dalle_pytorch.reversible import ReversibleSequence, SequentialSequence
from dalle_pytorch.attention import Attention, SparseAttention

# helpers

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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
        noncausal_attn_len = 0,
        sparse_attn = False,
        sparse_attn_global_indices = []
    ):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)

        for _, sparse_attn in zip(range(depth), sparse_layer):
            attn_class = Attention if not sparse_attn else partial(SparseAttention, sparse_attn_global_indices = sparse_attn_global_indices)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn_class(dim, causal = causal, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, noncausal_attn_len = noncausal_attn_len)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}

        self.layers = execute_type(layers, args_route = attn_route_map)

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)
