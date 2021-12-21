import torch
import torch.nn as nn

# helpers

def exists(val):
    return val is not None

class Cached(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *, cache=None, cache_key=None, **kwargs):
        return self.fn(x, **kwargs)

class FixCacheKey(nn.Module):
    def __init__(self, cache_key, fn):
        super().__init__()
        self.cache_key = cache_key
        self.fn = fn

    def forward(self, x, *, cache=None, **kwargs):
        return self.fn(x, cache=cache, cache_key=self.cache_key, **kwargs)
