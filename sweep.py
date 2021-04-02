
import argparse
from random import choice
from pathlib import Path

# torch

import torch
from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

# vision imports

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import deepspeed_utils
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE
from dalle_pytorch.simple_tokenizer import tokenize, tokenizer, VOCAB_SIZE

# argument parsing

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required = False)

group.add_argument('--vae_path', type = str,
                    help='path to your trained discrete VAE')

group.add_argument('--dalle_path', type = str,
                    help='path to your partially trained DALL-E')

parser.add_argument('--image_text_folder', type = str, required = True,
                    help='path to your folder of images and text for learning the DALL-E')

parser.add_argument('--truncate_captions', dest='truncate_captions',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')

parser.add_argument('--taming', dest='taming', action='store_true')

group.add_argument('--dalle_path', type = str, help='path to your partially trained DALL-E')

parser.add_argument('--learning_rate', type = float, default = 3e-4)


parser.add_argument('--text_seq_len', type = int, default = 256)
parser.add_argument('--depth', type = int, default = 4)
parser.add_argument('--heads', type = int, default = 4)
parser.add_argument('--dim_head', type = int, default = 64)
parser.add_argument('--reversible', type = bool, default = True)
parser.add_argument('--loss_img_weight', type = int, default = 7)

parser.add_argument('--attn_dropout', type = float, default = 0.1)
parser.add_argument('--ff_dropout', type = float, default = 0.1)

parser.add_argument('--epochs', type = int, default = 1)
parser.add_argument('--batch_size', type = int, default = 8)

parser.add_argument('--grad_clip_norm', type = float, default = 0.5)

parser.add_argument('--grad_clip_norm', type = float, default = 0.5)


parser.add_argument('--optimizer',
  default='adam',
  const='adam',
  nargs='?',
  choices=('adam', 'adamw')
  help='Use adam or adamw: (default: %(default)s)'
 )

parser.add_argument('--attn_types',
  default='full',
  const='full',
  nargs='?',
  choices=('full', 'axial_row', 'axial_col', 'conv_like'),
  help='Attention types to use: (default: %(default)s)'
 )

parser = deepspeed_utils.wrap_arg_parser(parser)

args = parser.parse_args()

# helpers

def exists(val):
    return val is not None

# constants

VAE_PATH = args.vae_path
DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
GRAD_CLIP_NORM = args.grad_clip_norm

MODEL_DIM = 512
TEXT_SEQ_LEN = args.text_seq_len
DEPTH = args.depth
HEADS = args.heads
DIM_HEAD = args.dim_head
REVERSIBLE = args.reversible
LOSS_IMG_WEIGHT = args.loss_img_weight
OPTIMIZER = args.optimizer
LR_DECAY = args.lr_decay

# initialize deepspeed

deepspeed_utils.init_deepspeed(args.deepspeed)

# reconstitute vae

if RESUME:
    dalle_path = Path(DALLE_PATH)
    assert dalle_path.exists(), 'DALL-E model file does not exist'

    loaded_obj = torch.load(str(dalle_path), map_location='cpu')

    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']

    if vae_params is not None:
        vae = DiscreteVAE(**vae_params)
    else:
        vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
        vae = vae_klass()
        
    dalle_params = dict(        
        **dalle_params
    )
    IMAGE_SIZE = vae.image_size
else:
    if exists(VAE_PATH):
        vae_path = Path(VAE_PATH)
        assert vae_path.exists(), 'VAE model file does not exist'

        loaded_obj = torch.load(str(vae_path))

        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

        vae = DiscreteVAE(**vae_params)
        vae.load_state_dict(weights)
    else:
        if deepspeed_utils.is_root_worker():
            print('using pretrained VAE for encoding images to tokens')
        vae_params = None

        vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
        vae = vae_klass()

    IMAGE_SIZE = vae.image_size

    dalle_params = dict(
        num_text_tokens = VOCAB_SIZE,
        text_seq_len = args.text_seq_len,
        dim = MODEL_DIM,
        depth = args.depth,
        heads = args.heads,
        dim_head = args.dim_head,
        reversible = args.reversible,
        loss_img_weight = args.loss_img_weight,
        attn_dropout = args.attn_dropout,
        ff_dropout = args.ff_dropout,
        attn_types = args.attn_types,
    )

# helpers

def save_model(path):
    if not deepspeed_utils.is_root_worker():
        return

    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'weights': dalle.state_dict()
    }

    torch.save(save_obj, path)

def group_weight(model):
    group_decay, group_no_decay = [], []
    for params in model.named_parameters():
        if 'transformer' in params[0]:
            if 'bias' in params[0] or 'norm' in params[0]:
                group_no_decay.append(params[1])
                continue
        group_decay.append(params[1])

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

# dataset loading

class TextImageDataset(Dataset):
    def __init__(self, folder, text_len = 256, image_size = 128):
        super().__init__()
        path = Path(folder)

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
        self.text_len = text_len

        self.image_tranform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size, scale = (0.6, 1.), ratio = (1., 1.)),
            T.ToTensor()
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

        tokenized_text = tokenize(description, self.text_len, truncate_text=args.truncate_captions).squeeze(0)
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
if deepspeed_utils.is_root_worker():
    print(f'{len(ds)} image-text pairs found for training')

dl = DataLoader(ds, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

# initialize DALL-E

dalle = DALLE(vae = vae, **dalle_params).cuda()

if RESUME:
    dalle.load_state_dict(weights)

# optimizer

if OPTIMIZER is 'adamw':
    opt = AdamW(group_weight(dalle), lr = LEARNING_RATE, betas = (0.9, 0.96), eps = 1e-08, weight_decay = 4.5e-2, amsgrad = False)
else:
    opt = Adam(dalle.parameters(), lr = LEARNING_RATE)

if LR_DECAY:
    scheduler = ReduceLROnPlateau(
        opt,
        mode = "min",
        factor = 0.5,
        patience = 10,
        cooldown = 10,
        min_lr = 1e-6,
        verbose = True,
    )

if deepspeed_utils.is_root_worker():
    # experiment tracker

    import wandb

    model_config = dict(
        depth = DEPTH,
        heads = HEADS,
        dim_head = DIM_HEAD
    )

    run = wandb.init(
        project = 'dalle_train_transformer',
        resume = RESUME,
        config = model_config,
    )
    torch.cuda.empty_cache()

# distribute

deepspeed_utils.check_batch_size(BATCH_SIZE)
deepspeed_config = {'train_batch_size': BATCH_SIZE}

(distr_dalle, opt, dl, scheduler) = deepspeed_utils.maybe_distribute(
    args=args,
    model=dalle,
    optimizer=opt,
    model_parameters=dalle.parameters(),
    training_data=ds if args.deepspeed else dl,
    lr_scheduler=scheduler if LR_DECAY else None,
    config_params=deepspeed_config,
)

# training

for epoch in range(EPOCHS):
    for i, (text, images, mask) in enumerate(dl):
        text, images, mask = map(lambda t: t.cuda(), (text, images, mask))

        loss = distr_dalle(text, images, mask = mask, return_loss = True)

        if args.deepspeed:
            distr_dalle.backward(loss)
        else:
            loss.backward()

        clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)

        if args.deepspeed:
            distr_dalle.step()
            # Gradients are automatically zeroed after the step
        else:
            opt.step()
            opt.zero_grad()

        if deepspeed_utils.is_root_worker():
            log = {}

            if i % 10 == 0:
                torch.cuda.empty_cache()
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

                save_model(f'./dalle.pt')
                wandb.save(f'./dalle.pt')

                log = {
                    **log,
                    'image': wandb.Image(image, caption = decoded_text)
                }

            wandb.log(log)

    if LR_DECAY:
        scheduler.step(loss)

    if deepspeed_utils.is_root_worker():
        # save trained model to wandb as an artifact every epoch's end

        model_artifact = wandb.Artifact('trained-dalle', type = 'model', metadata = dict(model_config))
        model_artifact.add_file('dalle.pt')
        run.log_artifact(model_artifact)

if deepspeed_utils.is_root_worker():
    save_model(f'./dalle-final.pt')
    wandb.save('./dalle-final.pt')
    model_artifact = wandb.Artifact('trained-dalle', type = 'model', metadata = dict(model_config))
    model_artifact.add_file('dalle-final.pt')
    run.log_artifact(model_artifact)

    wandb.finish()
