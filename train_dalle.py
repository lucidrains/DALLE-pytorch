import argparse
from random import choice
from pathlib import Path

# torch

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# vision imports

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import DiscreteVAE, DALLE
from dalle_pytorch.simple_tokenizer import tokenize, tokenizer, VOCAB_SIZE

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
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 0.5

MODEL_DIM = 512
TEXT_SEQ_LEN = 256
DEPTH = 2
HEADS = 4
DIM_HEAD = 64

IMAGE_SIZE = vae.image_size

# dataset loading

class TextImageDataset(Dataset):
    def __init__(self, folder, text_len = 256, image_size = 128):
        super().__init__()
        path = Path(args.image_text_folder)

        text_files = [*path.glob('**/*.txt')]

        image_files = [
            *path.glob('**/*.png'),
            *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg')
        ]

        text_files = {t.stem: t for t in text_files}
        image_files = {i.stem: i for i in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}

        self.image_tranform = T.Compose([
            T.CenterCrop(image_size),
            T.Resize(image_size),
            T.ToTensor(),
            T.Lambda(lambda t: t.expand(3, -1, -1)),
            T.Normalize((0.5,) * 3, (0.5,) * 3)
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]

        image = Image.open(image_file)
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        description = choice(descriptions)

        tokenized_text = tokenize(description).squeeze(0)
        mask = tokenized_text != 0

        image_tensor = self.image_tranform(image)
        return tokenized_text, image_tensor, mask

# create dataset and dataloader

ds = TextImageDataset(
    args.image_text_folder,
    text_len = TEXT_SEQ_LEN,
    image_size = IMAGE_SIZE
)

assert len(ds) > 0, 'dataset is empty'

dl = DataLoader(ds, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

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

# optimizer

opt = Adam(dalle.parameters(), lr = LEARNING_RATE)

# experiment tracker

import wandb
wandb.init(project = 'dalle_train_transformer')

# training

for epoch in range(EPOCHS):
    for i, (text, images, mask) in enumerate(dl):
        text, images, mask = map(lambda t: t.cuda(), (text, images, mask))

        loss = dalle(text, images, mask = mask, return_loss = True)

        loss.backward()
        clip_grad_norm_(dalle.parameters(), GRAD_CLIP_NORM)

        opt.step()
        opt.zero_grad()

        log = {}

        if i % 10 == 0:
            print(epoch, i, f'loss - {loss.item()}')

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item()
            }

        if i % 100 == 0:
            sample_text = text[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            image = dalle.generate_images(
                text[:1],
                mask = mask[:1],
                filter_thres = 0.9    # topk sampling at 0.9
            )

            torch.save(dalle.state_dict(), f'./dalle.pt')

            log = {
                **log,
                'image': wandb.Image(image, caption = decoded_text)
            }

        wandb.log(log)

torch.save(dalle.state_dict(), f'./dalle.pt')
wandb.finish()
