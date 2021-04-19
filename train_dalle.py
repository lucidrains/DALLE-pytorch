import argparse
from pathlib import Path

import torch

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import argparse

from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch import distributed_utils
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=False)

group.add_argument('--vae_path', type = str,
                    help='path to your trained discrete VAE')

group.add_argument('--dalle_path', type = str,
                    help='path to your partially trained DALL-E')

parser.add_argument('--image_text_folder', type = str, required = True,
    help='path to your folder of images and text for learning the DALL-E')

parser.add_argument('--truncate_captions', dest='truncate_captions', action = 'store_true',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')

parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type = float, default = 0.75,
                    help='Random resized crop lower ratio')

parser.add_argument('--chinese', dest='chinese', action = 'store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

parser.add_argument('--hug', dest='hug', action = 'store_true')

parser.add_argument('--bpe_path', type = str,
                    help='path to your huggingface BPE json file')

parser.add_argument('--fp16', action='store_true',
                    help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')

parser.add_argument('--wandb_name', default='dalle_train_transformer',
                    help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')
parser = distributed_utils.wrap_arg_parser(parser)
args = parser.parse_args()

# quit early if you used the wrong folder name
assert Path(args.image_text_folder).exists(), f'The path {args.image_text_folder} was not found.'

# helpers

def exists(val):
    return val is not None


# constants

VAE_PATH = args.vae_path
DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH)

EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 0.5

MODEL_DIM = 512
TEXT_SEQ_LEN = 256
DEPTH = 2
HEADS = 4
DIM_HEAD = 64
REVERSIBLE = True
LOSS_IMG_WEIGHT = 7
LR_DECAY = False

# initialize distributed backend

distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

using_deepspeed = \
    distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

# tokenizer

if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

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
        if distr_backend.is_root_worker():
            print('using pretrained VAE for encoding images to tokens')
        vae_params = None

        vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
        vae = vae_klass()

    IMAGE_SIZE = vae.image_size

    dalle_params = dict(
        num_text_tokens=tokenizer.vocab_size,
        text_seq_len=TEXT_SEQ_LEN,
        dim=MODEL_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        reversible=REVERSIBLE,
        loss_img_weight=LOSS_IMG_WEIGHT
    )

# configure OpenAI VAE for float16s

if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
    vae.enc.blocks.output.conv.use_float16 = True


# helpers

def save_model(path):
    if not distr_backend.is_root_worker():
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


# create dataset and dataloader

is_shuffle = distributed_utils.using_backend(distributed_utils.HorovodBackend)

ds = TextImageDataset(
    args.image_text_folder,
    text_len=TEXT_SEQ_LEN,
    image_size=IMAGE_SIZE,
    resize_ratio=args.resize_ratio,
    truncate_captions=args.truncate_captions,
    tokenizer=tokenizer,
    shuffle=is_shuffle,
)

assert len(ds) > 0, 'dataset is empty'
if distr_backend.is_root_worker():
    print(f'{len(ds)} image-text pairs found for training')

if is_shuffle:
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=distr_backend.get_world_size(),
        rank=distr_backend.get_rank()
    )
else:
    data_sampler = None

dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_shuffle, drop_last=True, sampler=data_sampler)

# initialize DALL-E

dalle = DALLE(vae=vae, **dalle_params)
if args.fp16:
    dalle = dalle.half()
dalle = dalle.cuda()

if RESUME:
    dalle.load_state_dict(weights)

# optimizer

opt = Adam(dalle.parameters(), lr=LEARNING_RATE)

if LR_DECAY:
    scheduler = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=10,
        cooldown=10,
        min_lr=1e-6,
        verbose=True,
    )

if distr_backend.is_root_worker():
    # experiment tracker

    import wandb

    model_config = dict(
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD
    )

    run = wandb.init(
        project=args.wandb_name,
        resume=RESUME,
        config=model_config,
    )

# distribute

distr_backend.check_batch_size(BATCH_SIZE)
deepspeed_config = {
    'train_batch_size': BATCH_SIZE,
    'gradient_clipping': GRAD_CLIP_NORM,
    'fp16': {
        'enabled': args.fp16,
    },
}

(distr_dalle, distr_opt, distr_dl, distr_scheduler) = distr_backend.distribute(
    args=args,
    model=dalle,
    optimizer=opt,
    model_parameters=dalle.parameters(),
    training_data=ds if using_deepspeed else dl,
    lr_scheduler=scheduler if LR_DECAY else None,
    config_params=deepspeed_config,
)
avoid_model_calls = using_deepspeed and args.fp16

# training
# Avoid allocation error due to potential bug in deepspeed.
# See https://github.com/lucidrains/DALLE-pytorch/issues/161
for epoch in range(EPOCHS):
    for i, (text, images) in enumerate(distr_dl):
        if args.fp16:
            images = images.half()
        text, images = map(lambda t: t.cuda(), (text, images))

        loss = distr_dalle(text, images, return_loss=True)

        if using_deepspeed:
            distr_dalle.backward(loss)
            distr_dalle.step()
            # Gradients are automatically zeroed after the step
        else:
            loss.backward()
            clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)
            distr_opt.step()
            distr_opt.zero_grad()

        # Collective loss, averaged
        avg_loss = distr_backend.average_all(loss)

        if distr_backend.is_root_worker():
            log = {}

            if i % 10 == 0:
                print(epoch, i, f'loss - {avg_loss.item()}')

                log = {
                    **log,
                    'epoch': epoch,
                    'iter': i,
                    'loss': avg_loss.item()
                }

            if i % 100 == 0:
                sample_text = text[:1]
                token_list = sample_text.masked_select(sample_text != 0).tolist()
                decoded_text = tokenizer.decode(token_list)

                if not avoid_model_calls:
                    # CUDA index errors when we don't guard this
                    image = dalle.generate_images(text[:1], filter_thres=0.9)  # topk sampling at 0.9

                save_model(f'./dalle.pt')
                wandb.save(f'./dalle.pt')

                log = {
                    **log,
                }
                if not avoid_model_calls:
                    log['image'] = wandb.Image(image, caption=decoded_text)

            wandb.log(log)

    if LR_DECAY:
        distr_scheduler.step(loss)

    if distr_backend.is_root_worker():
        # save trained model to wandb as an artifact every epoch's end

        model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
        model_artifact.add_file('dalle.pt')
        run.log_artifact(model_artifact)

if distr_backend.is_root_worker():
    save_model(f'./dalle-final.pt')
    wandb.save('./dalle-final.pt')
    model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
    model_artifact.add_file('dalle-final.pt')
    run.log_artifact(model_artifact)

    wandb.finish()
