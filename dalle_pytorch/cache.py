import torch.nn as nn

class FixCacheKey(nn.Module):
    def __init__(self, cache_key, fn):
        super().__init__()
        self.cache_key = cache_key
        self.fn = fn

    def forward(self, x, *, cache=None, **kwargs):
        return self.fn(x, cache=cache, cache_key=self.cache_key, **kwargs)
