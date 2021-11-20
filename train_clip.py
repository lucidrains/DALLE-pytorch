import argparse
import os
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

import wandb
from dalle_pytorch import CLIP, distributed_utils
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer  # TODO support different tokenizers

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', type=str, required=True, help= 'path to your folder of images for learning the discrete VAE and its codebook')
parser = distributed_utils.wrap_arg_parser(parser)
train_group = parser.add_argument_group('Training settings')
train_group.add_argument('--epochs', type=int, default=20, help='number of epochs')
train_group.add_argument('--batch_size', type=int, default=32, help='batch size')
train_group.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
train_group.add_argument('--lr_decay', type=bool, default=True, help='whether to decay learning rate')
train_group.add_argument('--clip_grad_norm', type=float, default=1.0, help='gradient norm clipping')
train_group.add_argument('--resize_ratio', type=float, default=0.75, help='resize ratio')
train_group.add_argument('--truncate_captions', type=bool, default=False, help='truncate captions to max length')
train_group.add_argument('--save_every_n_steps', '--save_frequency', type=int, default=1000, help='save model every n steps')
train_group.add_argument('--log_frequency', type=int, default=10, help='log every n steps')
# train_group.add_argument('--clip_output_file_name', type=str, default='clip_latest.pt', help='clip output file name') # TODO 
train_group.add_argument('--num_workers', type=int, default=1, help='number of workers')
train_group.add_argument('--checkpoint_dir', type=str, default='model.pt', help='path to save model')

distributed_group = parser.add_argument_group('Distributed settings')
distributed_group.add_argument('--gradient_accumulation_steps', '--ga_steps', type=int, default=1, help='number of gradient accumulation steps, increase if using deepspeed stage 3 round robin')

model_group = parser.add_argument_group('Model settings')
model_group.add_argument('--dim_text', type=int, default=512, help='text embedding dimension')
model_group.add_argument('--dim_image', type=int, default=512, help='image embedding dimension')
model_group.add_argument('--dim_latent', type=int, default=512, help='latent dimension')
model_group.add_argument('--text_enc_depth', type=int, default=6, help='text encoder depth')
model_group.add_argument('--text_seq_len', type=int, default=256, help='text sequence length')
model_group.add_argument('--text_heads', type=int, default=8, help='text multihead attention heads')
model_group.add_argument('--num_visual_tokens', type=int, default=512, help='number of tokens in the visual codebook')
model_group.add_argument('--visual_enc_depth', type=int, default=6, help='visual encoder depth')
model_group.add_argument('--visual_heads', type=int, default=8, help='visual multihead attention heads')
model_group.add_argument('--visual_image_size', type=int, default=256, help='visual image size')
model_group.add_argument('--visual_patch_size', type=int, default=32, help='visual patch size')
model_group.add_argument('--channels', type=int, default=3, help='number of channels')
model_group.add_argument('--fp16', type=bool, default=False, help='use fp16') # TODO works very well with deepspeed, need to test with raw torch

# TODO presently hardcoded to the number used in SimpleTokenizer - need to make this more flexible
# model_group.add_argument('--num_text_tokens', type=int, default=, help='number of tokens in the text codebook')

args = parser.parse_args()

CHECKPOINT_DIR = Path(args.checkpoint_dir)

# constants

SAVE_EVERY_N_STEPS = args.save_every_n_steps
LOG_FREQUENCY = args.log_frequency
CLIP_OUTPUT_FILE_NAME = args.clip_output_file_name

DATASET_DIR = args.dataset_dir

EPOCHS = args.epochs
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
LR_DECAY = args.lr_decay
RESIZE_RATIO = args.resize_ratio
TRUNCATE_CAPTIONS = args.truncate_captions
NUM_WORKERS = args.num_workers

GRAD_CLIP_NORM = args.clip_grad_norm
DIM_TEXT = args.dim_text
DIM_IMAGE = args.dim_image
DIM_LATENT = args.dim_latent
# NUM_TEXT_TOKENS = args.num_text_tokens
NUM_TEXT_TOKENS=49408 # TODO make this more flexible
TEXT_ENC_DEPTH = args.text_enc_depth
TEXT_SEQ_LEN = args.text_seq_len
TEXT_HEADS = args.text_heads
NUM_VISUAL_TOKENS = args.num_visual_tokens
VISUAL_ENC_DEPTH = args.visual_enc_depth
VISUAL_HEADS = args.visual_heads
VISUAL_IMAGE_SIZE = args.visual_image_size
VISUAL_PATCH_SIZE = args.visual_patch_size
CHANNELS = args.channels

# initialize device for tensor operations (pytorch only)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IS_FP16 = args.fp16 
# TODO would using sandwich norm for the transformers in CLIP help with stability issues? 
# TODO otherwise deepspeed automatic nan skipping seems to stabilize training after about 50 steps

# initialize distributed backend

distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

using_deepspeed = distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

# initialize logger with model params

clip_model_params = dict(
    dim_text=DIM_TEXT,
    dim_image=DIM_IMAGE,
    dim_latent=DIM_LATENT,
    num_text_tokens=NUM_TEXT_TOKENS,
    text_enc_depth=TEXT_ENC_DEPTH,
    text_seq_len=TEXT_SEQ_LEN,
    text_heads=TEXT_HEADS,
    num_visual_tokens=NUM_VISUAL_TOKENS,
    visual_enc_depth=VISUAL_ENC_DEPTH,
    visual_heads=VISUAL_HEADS,
    visual_image_size=VISUAL_IMAGE_SIZE,
    visual_patch_size=VISUAL_PATCH_SIZE,
    channels=CHANNELS)

if distr_backend.is_root_worker():
    model_config = dict(**clip_model_params, **dict(
        fp16=IS_FP16,
        learning_rate=LEARNING_RATE,
        grad_clip_norm=GRAD_CLIP_NORM,
        num_workers=NUM_WORKERS,
        resize_ratio=RESIZE_RATIO,
        truncate_captions=TRUNCATE_CAPTIONS,
        log_frequency=LOG_FREQUENCY,
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        device=DEVICE))
    run = wandb.init(project='dalle_train_clip', job_type='train_model', config=model_config) # TODO add arg for project name


# data
is_shuffled = True # TODO - use distributed_utils to disable shuffling for horovod

ds = TextImageDataset(
    DATASET_DIR,
    text_len=TEXT_SEQ_LEN,
    image_size=VISUAL_IMAGE_SIZE,
    resize_ratio=RESIZE_RATIO,
    truncate_captions=TRUNCATE_CAPTIONS,
    tokenizer=tokenizer, # TODO support different tokenizers
    shuffle=is_shuffled, # TODO doesnt work with horovod
)
assert len(ds) > 0, 'dataset is empty'
dl = DataLoader(ds, BATCH_SIZE, shuffle=is_shuffled, num_workers=NUM_WORKERS, pin_memory=True)
if distr_backend.is_root_worker():
    print(f'{len(ds)} images found for training')

clip_model = CLIP(**clip_model_params)
if not using_deepspeed: # TODO test this works _without_ deepspeed
    clip_model = clip_model.to(DEVICE)

# optimizer

# TODO these defaults were selected from the paper for the ViT-32, may not be optimal for user datasets. 
# I believe they also use weight decay wheras this doesnt.
opt = torch.optim.Adam(clip_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-6) 

# scheduler

if not using_deepspeed: # TODO test this works _without_ deepspeed
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
else:
    sched = None


# distribute

# precalculate the number of steps per epoch, needed by deepspeed which assumes a single epoch (per worker) regime
distr_backend.check_batch_size(BATCH_SIZE)
batches_per_epoch = len(ds) // BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
batches_total = batches_per_epoch * EPOCHS
# TODO printout here would be nice

deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "round_robin_gradients": True, # This settings seems to scale very well in both the distributed scenario and when using gradient accumulation on a single GPU. 
        "offload_param": {
            "device": "cpu", # The NVM-e option from deepspeed infinity is quite useful but presently doesnt work with checkpointing.
        },
        "offload_optimizer": {
            "device": "cpu", # The NVM-e option from deepspeed infinity is quite useful but presently doesnt work with checkpointing.
        },
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 1e-6,
            "warmup_max_lr": LEARNING_RATE,
            "warmup_num_steps": len(dl) * EPOCHS // BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS, # num grad accum steps
            "total_num_steps": batches_total,
        }
    },
    'gather_fp16_weights_on_model_save': True, # TODO have not tested this on a distributed setup where it could cause issues. Saves tons of disk space however.
    'train_batch_size': BATCH_SIZE,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    'gradient_clipping': GRAD_CLIP_NORM,
    'fp16': {
        'enabled': IS_FP16,
        'initial_scale_power': 32, # the default, often it's better to start lower around 16-24
    },
    'wall_clock_breakdown': False, # TODO enable per cli
}

(distr_clip, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
    args=args,
    model=clip_model,
    optimizer=opt,
    model_parameters=clip_model.parameters(),
    training_data=ds if using_deepspeed else dl,
    lr_scheduler=sched if not using_deepspeed else None, # sched if not using_deepspeed else None,
    config_params=deepspeed_config,
)

# Scheduler 
using_deepspeed_sched = False
if distr_sched is None:
    distr_sched = sched
elif using_deepspeed:
    using_deepspeed_sched = True


if LR_DECAY and distr_sched is None and sched is not None:
    # Prefer scheduler in `deepspeed_config` if available
    distr_scheduler = sched

def save_model(path, model_params, model): # TODO - deepspeed doesnt save the hyperparams in its checkpoints, consider warning user about this.
    save_obj = {
        'hparams': model_params,
    }
    if using_deepspeed:
        cp_path = Path(path)
        path_sans_extension = cp_path.parent / cp_path.stem
        cp_dir = str(path_sans_extension) + '_ds_cp'

        distr_clip.save_checkpoint(cp_dir, client_state=save_obj)
        # We do not return so we do get a "normal" checkpoint to refer to.
        # TODO auxiliary checkpointing from @janEbert, needs to be implemented.

    if not distr_backend.is_root_worker():
        return
    save_obj = {**save_obj, 'weights': model.state_dict()}
    torch.save(save_obj, path)

ENABLE_WEBDATASET = False # TODO hardcoded for now

for epoch in range(0, EPOCHS):
    # if data_sampler: data_sampler.set_epoch(epoch) # TODO - implement this for horovod
    # for i, (text, images) in enumerate((dl if ENABLE_WEBDATASET else distr_dl)): # TODO implement webdataset
    # TODO - implement samples per second calculation (can also use wallclock time from deepspeed)
    for i, (text, images) in enumerate(dl):
        if args.fp16: images = images.half()
        text, images = map(lambda t: t.to(DEVICE), (text, images))
        mask = torch.ones_like(text).bool().to(DEVICE)
        loss = distr_clip(text, images, text_mask=mask, return_loss=True)
        if using_deepspeed:
            distr_clip.backward(loss) # propagate gradients across all workers
            distr_clip.step() # use opt/config stored in distr_clip to clip gradients, update params and reset gradients for next iteration
        else:
            loss.backward() # propogate error backwards with autograd
            clip_grad_norm_(distr_clip.parameters(), GRAD_CLIP_NORM) # clip gradients to prevent exploding gradients
            distr_opt.step() # update parameters
            distr_opt.zero_grad() # zero gradients to be ready for next iteration
        
        # Collective loss, averaged
        avg_loss = distr_backend.average_all(loss)

        log = {}

        if i % LOG_FREQUENCY == 0 and distr_backend.is_root_worker():
            print(epoch, i, f'loss - {avg_loss.item():.12f}')
            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': avg_loss.item()
            }

        if distr_backend.is_root_worker(): wandb.log(log)

        # TODO large batch sizes make this far too frequent and DeepSpeed saves a global step number on the epoch in any case. Should we remove it?
        if i % SAVE_EVERY_N_STEPS == 0:
            save_model(f'{epoch}_{i}.pt', clip_model_params, clip_model)
    save_model(f'checkpoints/epoch_{epoch}.pt', clip_model_params, clip_model)