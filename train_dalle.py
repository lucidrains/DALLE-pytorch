import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from dalle_pytorch import DiscreteVAE, DALLE
from dalle_pytorch.simple_tokenizer import tokenize, VOCAB_SIZE

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--vae_path', type = str, required = True,
                    help='path to your trained discrete VAE')

parser.add_argument('--image_text_folder', type = str, required = True,
                    help='path to your folder of images and text for learning the DALL-E')

args = parser.parse_args()

# reconstitute vae

vae_path = Path(args.vae_path)
assert vae_path.exists(), 'VAE model file does not exist'

loaded_obj = torch.load(str(vae_path))

hparams, weights = loaded_obj['hparams'], loaded_obj['weights']

vae = DiscreteVAE(**hparams)
vae.load_state_dict(weights)

# constants

EPOCHS = 20
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 0.5

MODEL_DIM = 512
TEXT_SEQ_LEN = 256
DEPTH = 2
HEADS = 4
DIM_HEAD = 64

IMAGE_SIZE = vae.image_size

# dataset loading

## todo

# initialize DALL-E

dalle = DALLE(
    vae = vae,
    num_text_tokens = VOCAB_SIZE,
    text_seq_len = TEXT_SEQ_LEN,
    dim = MODEL_DIM,
    depth = DEPTH,
    heads = HEADS,
    dim_head = DIM_HEAD
).cuda()

text = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TEXT_SEQ_LEN)).cuda()
images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()
mask = torch.ones_like(text).bool().cuda()

loss = dalle(text, images, mask = mask, return_loss = True)
loss.backward()

# experiment tracker

## todo

# training

## todo
