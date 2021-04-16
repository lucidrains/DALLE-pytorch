import PIL
import argparse
from pathlib import Path
from random import choice

import nonechucks as nc
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from dalle_pytorch import (DALLE, DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE1024, distributed_utils)
from dalle_pytorch.tokenizer import ChineseTokenizer, HugTokenizer, tokenizer


class TextImageDataset(Dataset):
    """ [TextImageDataset]: [expects images and txt files in same folder with same basename]

    Args:
    [folder]: [basedir of your image and text files]
    [text_len]: [tokenizer text_seq_length]
    [image_size]: [image size]
    [random_resize]: [center crops and zooms in by a random amount. `random_resize=1` disables zooming.]
    """    
    def __init__(self, folder, tokenizer, text_len = 256, image_size = 128, random_resize=0.6, truncate_text=False):
        super().__init__()
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]

        image_files = [
            *path.glob('**/*.png'),
            *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'),
            *path.glob('**/*.bmp')
        ]

        text_files = {t.stem: t for t in text_files}
        image_files = {i.stem: i for i in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len

        self.tokenizer = tokenizer
        self.image_tranform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size, scale = (random_resize, 1.), ratio = (1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)
    
    def attempt_tokenize(self):
        try:
            tokenized_text = self.tokenizer.tokenize(description, self.text_len, truncate_text=args.truncate_captions).squeeze(0)
        except Exception as ex:
            return None
        

    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]

        try:
            image = Image.open(image_file)
            image_tensor = self.image_tranform(image)
        except (PIL.UnidentifiedImageError, OSError) as ex:
            short_msg = ex[:60]
            print(f"Image {image_file} corrupt - dropping image from dataset. {short_msg}...")
            return None
        
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        description = choice(descriptions)

        try:
            tokenized_text = self.tokenizer.tokenize(description, self.text_len, truncate_text=args.truncate_captions).squeeze(0)
        except RuntimeError as ex:
            print(f"Exception occurred during image - dropping image from dataset. {short_msg}...")
            return None

        return tokenized_text, image_tensor