import argparse
from pathlib import Path
from tqdm import tqdm

# torch

import torch

from einops import repeat

# vision imports

from PIL import Image
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE1024, DALLE
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, YttmTokenizer, ChineseTokenizer

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--dalle_path', type = str, required = True,
                    help='path to your trained DALL-E')

parser.add_argument('--text', type = str, required = True,
                    help='your text prompt')

parser.add_argument('--num_images', type = int, default = 128, required = False,
                    help='number of images')

parser.add_argument('--batch_size', type = int, default = 4, required = False,
                    help='batch size')

parser.add_argument('--top_k', type = float, default = 0.9, required = False,
                    help='top k filter threshold')

parser.add_argument('--outputs_dir', type = str, default = './outputs', required = False,
                    help='output directory')

parser.add_argument('--bpe_path', type = str,
                    help='path to your huggingface BPE json file')

parser.add_argument('--hug', dest='hug', action = 'store_true')

parser.add_argument('--chinese', dest='chinese', action = 'store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

args = parser.parse_args()

# helper fns

def exists(val):
    return val is not None

# tokenizer

if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

# load DALL-E

dalle_path = Path(args.dalle_path)

assert dalle_path.exists(), 'trained DALL-E must exist'

load_obj = torch.load(str(dalle_path))
dalle_params, vae_params, weights = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights')

dalle_params.pop('vae', None) # cleanup later

if vae_params is not None:
    vae = DiscreteVAE(**vae_params)
elif not args.taming:
    vae = OpenAIDiscreteVAE()
else:
    vae = VQGanVAE1024()


dalle = DALLE(vae = vae, **dalle_params).cuda()

dalle.load_state_dict(weights)

# generate images

image_size = vae.image_size

texts = args.text.split('|')

for text in tqdm(texts):
    text = tokenizer.tokenize([args.text], dalle.text_seq_len).cuda()

    text = repeat(text, '() n -> b n', b = args.num_images)

    outputs = []

    for text_chunk in tqdm(text.split(args.batch_size), desc = f'generating images for - {text}'):
        output = dalle.generate_images(text_chunk, filter_thres = args.top_k)
        outputs.append(output)

    outputs = torch.cat(outputs)

    # save all images

    outputs_dir = Path(args.outputs_dir) / args.text.replace(' ', '_')
    outputs_dir.mkdir(parents = True, exist_ok = True)

    for i, image in tqdm(enumerate(outputs), desc = 'saving images'):
        save_image(image, outputs_dir / f'{i}.jpg', normalize=True)

    print(f'created {args.num_images} images at "{str(outputs_dir)}"')
