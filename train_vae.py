import math
from math import sqrt
import argparse
from pathlib import Path

# torch

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle classes and utils

from dalle_pytorch import distributed_utils
from dalle_pytorch import DiscreteVAE

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--image_folder', type = str, required = True,
                    help='path to your folder of images for learning the discrete VAE and its codebook')

parser.add_argument('--image_size', type = int, required = False, default = 128,
                    help='image size')

parser = distributed_utils.wrap_arg_parser(parser)

args = parser.parse_args()

# constants

IMAGE_SIZE = args.image_size
IMAGE_PATH = args.image_folder

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
LR_DECAY_RATE = 0.98

NUM_TOKENS = 8192
NUM_LAYERS = 2
NUM_RESNET_BLOCKS = 2
SMOOTH_L1_LOSS = False
EMB_DIM = 512
HID_DIM = 256
KL_LOSS_WEIGHT = 0

STARTING_TEMP = 1.
TEMP_MIN = 0.5
ANNEAL_RATE = 1e-6

NUM_IMAGES_SAVE = 4

# initialize distributed backend

distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

using_deepspeed = \
    distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

# data

ds = ImageFolder(
    IMAGE_PATH,
    T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor()
    ])
)

if distributed_utils.using_backend(distributed_utils.HorovodBackend):
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=distr_backend.get_world_size(),
        rank=distr_backend.get_rank())
else:
    data_sampler = None

dl = DataLoader(ds, BATCH_SIZE, shuffle = not data_sampler, sampler=data_sampler)

vae_params = dict(
    image_size = IMAGE_SIZE,
    num_layers = NUM_LAYERS,
    num_tokens = NUM_TOKENS,
    codebook_dim = EMB_DIM,
    hidden_dim   = HID_DIM,
    num_resnet_blocks = NUM_RESNET_BLOCKS
)

vae = DiscreteVAE(
    **vae_params,
    smooth_l1_loss = SMOOTH_L1_LOSS,
    kl_div_loss_weight = KL_LOSS_WEIGHT
)
if not using_deepspeed:
    vae = vae.cuda()


assert len(ds) > 0, 'folder does not contain any images'
if distr_backend.is_root_worker():
    print(f'{len(ds)} images found for training')

# optimizer

opt = Adam(vae.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)


if distr_backend.is_root_worker():
    # weights & biases experiment tracking

    import wandb

    model_config = dict(
        num_tokens = NUM_TOKENS,
        smooth_l1_loss = SMOOTH_L1_LOSS,
        num_resnet_blocks = NUM_RESNET_BLOCKS,
        kl_loss_weight = KL_LOSS_WEIGHT
    )

    run = wandb.init(
        project = 'dalle_train_vae',
        job_type = 'train_model',
        config = model_config
    )

# distribute

distr_backend.check_batch_size(BATCH_SIZE)
deepspeed_config = {'train_batch_size': BATCH_SIZE}

(distr_vae, distr_opt, distr_dl, distr_sched) = distr_backend.distribute(
    args=args,
    model=vae,
    optimizer=opt,
    model_parameters=vae.parameters(),
    training_data=ds if using_deepspeed else dl,
    lr_scheduler=sched,
    config_params=deepspeed_config,
)

def save_model(path):
    save_obj = {
        'hparams': vae_params,
    }
    if using_deepspeed:
        cp_path = Path(path)
        path_sans_extension = cp_path.parent / cp_path.stem
        cp_dir = str(path_sans_extension) + '-ds-cp'

        distr_vae.save_checkpoint(cp_dir, client_state=save_obj)
        # We do not return so we do get a "normal" checkpoint to refer to.

    if not distr_backend.is_root_worker():
        return

    save_obj = {
        **save_obj,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)

# starting temperature

global_step = 0
temp = STARTING_TEMP

for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(distr_dl):
        images = images.cuda()

        loss, recons = distr_vae(
            images,
            return_loss = True,
            return_recons = True,
            temp = temp
        )

        if using_deepspeed:
            # Gradients are automatically zeroed after the step
            distr_vae.backward(loss)
            distr_vae.step()
        else:
            distr_opt.zero_grad()
            loss.backward()
            distr_opt.step()

        logs = {}

        if i % 100 == 0:
            if distr_backend.is_root_worker():
                k = NUM_IMAGES_SAVE

                with torch.no_grad():
                    codes = vae.get_codebook_indices(images[:k])
                    hard_recons = vae.decode(codes)

                images, recons = map(lambda t: t[:k], (images, recons))
                images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
                images, recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = True, range = (-1, 1)), (images, recons, hard_recons))

                logs = {
                    **logs,
                    'sample images':        wandb.Image(images, caption = 'original images'),
                    'reconstructions':      wandb.Image(recons, caption = 'reconstructions'),
                    'hard reconstructions': wandb.Image(hard_recons, caption = 'hard reconstructions'),
                    'codebook_indices':     wandb.Histogram(codes),
                    'temperature':          temp
                }

                wandb.save('./vae.pt')
            save_model(f'./vae.pt')

            # temperature anneal

            temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

            # lr decay

            if not using_deepspeed:
                # Scheduler is automatically progressed after the step
                # when using DeepSpeed.
                distr_sched.step()

        # Collective loss, averaged
        avg_loss = distr_backend.average_all(loss)

        if distr_backend.is_root_worker():
            if i % 10 == 0:
                lr = distr_sched.get_last_lr()[0]
                print(epoch, i, f'lr - {lr:6f} loss - {avg_loss.item()}')

                logs = {
                    **logs,
                    'epoch': epoch,
                    'iter': i,
                    'loss': avg_loss.item(),
                    'lr': lr
                }

            wandb.log(logs)
        global_step += 1

    if distr_backend.is_root_worker():
        # save trained model to wandb as an artifact every epoch's end

        model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
        model_artifact.add_file('vae.pt')
        run.log_artifact(model_artifact)

if distr_backend.is_root_worker():
    # save final vae and cleanup

    save_model('./vae-final.pt')
    wandb.save('./vae-final.pt')

    model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
    model_artifact.add_file('vae-final.pt')
    run.log_artifact(model_artifact)

    wandb.finish()
