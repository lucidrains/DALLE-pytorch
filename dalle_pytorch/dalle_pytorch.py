from math import log2
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helpers

def exists(val):
    return val is not None

# classes

class DiscreteVAE(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim = 512,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, num_tokens, 1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1)
        )

        self.codebook = nn.Embedding(num_tokens, dim)

    def forward(
        self,
        img,
        return_recon_loss = False
    ):
        logits = self.encoder(img)
        soft_one_hot = F.gumbel_softmax(logits, tau = 1.)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_recon_loss:
            return out

        loss = F.mse_loss(img, out)
        return loss

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
