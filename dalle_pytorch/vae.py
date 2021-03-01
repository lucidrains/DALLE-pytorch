import io
import sys
import os, sys
import requests
import PIL
import warnings
import os
import hashlib
import urllib
from tqdm import tqdm
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

# constants

ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'
DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'
EPS = 0.1

# helpers methods

def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location = torch.device('cpu'))

def map_pixels(x):
    return (1 - 2 * EPS) * x + EPS

def unmap_pixels(x):
    return torch.clamp((x - EPS) / (1 - 2 * EPS), 0, 1)

def download(url, root=os.path.expanduser("~/.cache/dalle")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)
    download_target_tmp = os.path.join(root, f'tmp.{filename}')

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    os.rename(download_target_tmp, download_target)
    return download_target

# adapter class

class OpenAIDiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import dall_e
        except:
            print(f'you need to "pip install git+https://github.com/openai/DALL-E.git" before you can use the pretrained OpenAI Discrete VAE')
            sys.exit()

        self.enc = load_model(download(ENCODER_PATH))
        self.dec = load_model(download(DECODER_PATH))
        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc(img)
        z = torch.argmax(z_logits, dim = 1)
        return rearrange(z, 'b h w -> b (h w)')

    def decode(self, img_seq):
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(sqrt(n)))

        z = F.one_hot(img_seq, num_classes = self.num_tokens)
        z = rearrange(z, 'b h w c -> b c h w').float()
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, img):
        raise NotImplemented
