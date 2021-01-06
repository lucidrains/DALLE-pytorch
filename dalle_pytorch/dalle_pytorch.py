import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helpers

def exists(val):
    return val is not None

# main classes

class CLIP(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return x

class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_text_tokens,
        num_visual_tokens,
        depth
    ):
        super().__init__()

    def forward(self, x):
        return x
