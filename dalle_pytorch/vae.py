import io
import sys
import os, sys
import requests
import PIL
import warnings
import os
import hashlib
import urllib
import yaml
from pathlib import Path
from tqdm import tqdm
from math import sqrt
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from dalle_pytorch import distributed_utils

# constants

CACHE_PATH = os.path.expanduser("~/.cache/dalle")

OPENAI_VAE_ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'
OPENAI_VAE_DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'

VQGAN_VAE_PATH = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
VQGAN_VAE_CONFIG_PATH = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'

# helpers methods

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location = torch.device('cpu'))

def map_pixels(x, eps = 0.1):
    return (1 - 2 * eps) * x + eps

def unmap_pixels(x, eps = 0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)

def download(url, filename = None, root = CACHE_PATH):
    if (
            not distributed_utils.is_distributed
            or distributed_utils.backend.is_local_root_worker()
    ):
        os.makedirs(root, exist_ok = True)
    filename = default(filename, os.path.basename(url))

    download_target = os.path.join(root, filename)
    download_target_tmp = os.path.join(root, f'tmp.{filename}')

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if (
            distributed_utils.is_distributed
            and not distributed_utils.backend.is_local_root_worker()
            and not os.path.isfile(download_target)
    ):
        # If the file doesn't exist yet, wait until it's downloaded by the root worker.
        distributed_utils.backend.local_barrier()

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
    if (
            distributed_utils.is_distributed
            and distributed_utils.backend.is_local_root_worker()
    ):
        distributed_utils.backend.local_barrier()
    return download_target

# pretrained Discrete VAE from OpenAI

class OpenAIDiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = load_model(download(OPENAI_VAE_ENCODER_PATH))
        self.dec = load_model(download(OPENAI_VAE_DECODER_PATH))

        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc.blocks(img)
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

# VQGAN from Taming Transformers paper
# https://arxiv.org/abs/2012.09841

class VQGanVAE1024(nn.Module):
    def __init__(self):
        super().__init__()

        model_filename = 'vqgan.1024.model.ckpt'
        config_filename = 'vqgan.1024.config.yml'

        download(VQGAN_VAE_CONFIG_PATH, config_filename)
        download(VQGAN_VAE_PATH, model_filename)

        config = OmegaConf.load(str(Path(CACHE_PATH) / config_filename))
        model = VQModel(**config.model.params)

        state = torch.load(str(Path(CACHE_PATH) / model_filename), map_location = 'cpu')['state_dict']
        model.load_state_dict(state, strict = False)

        self.model = model

        self.num_layers = 4
        self.image_size = 256
        self.num_tokens = 1024

        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameters(
            self, self.model.quantize.embedding.weight)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        return rearrange(indices, '(b n) () -> b n', b = b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented
