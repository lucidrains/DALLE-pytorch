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
        assert exists(cache) and exists(cache_key)  # dbg

        if exists(cache) and cache_key in cache:
            prefix = cache[cache_key]
            assert prefix.shape[1] + 1 == x.shape[1], f'{prefix.shape[1]} {x.shape[1]} {cache_key} {cache.keys()}'  # TODO: Change to <= for prod
            suffix = self.fn(x[:, prefix.shape[1]:, :], **kwargs)
            out = torch.cat([prefix, suffix], dim=1)
        else:
            out = self.fn(x, **kwargs)

        if exists(cache):
            cache[cache_key] = out
        return out

class FixCacheKey(nn.Module):
    def __init__(self, cache_key, fn):
        super().__init__()
        self.cache_key = cache_key
        self.fn = fn

    def forward(self, x, *, cache=None, **kwargs):
        return self.fn(x, cache=cache, cache_key=self.cache_key, **kwargs)
