import math
from math import sqrt
import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle classes and utils

from dalle_pytorch import deepspeed_utils
from dalle_pytorch import DiscreteVAE

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--image_folder', type = str, required = True,
                    help='path to your folder of images for learning the discrete VAE and its codebook')

parser.add_argument('--image_size', type = int, required = False, default = 128,
                    help='image size')

parser = deepspeed_utils.wrap_arg_parser(parser)

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

dl = DataLoader(ds, BATCH_SIZE, shuffle = True)

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
).cuda()


assert len(ds) > 0, 'folder does not contain any images'
print(f'{len(ds)} images found for training')

def save_model(path):
    if args.local_rank > 0:
        return

    save_obj = {
        'hparams': vae_params,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)

# optimizer

opt = Adam(vae.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)

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

# distribute with deepspeed

deepspeed_config = {'train_batch_size': BATCH_SIZE}

(distr_vae, opt, dl, _) = deepspeed_utils.maybe_init_deepspeed(
    args=args,
    model=vae,
    optimizer=opt,
    model_parameters=vae.parameters(),
    training_data=ds if args.deepspeed else dl,
    config_params=deepspeed_config,
)

# starting temperature

global_step = 0
temp = STARTING_TEMP

for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(dl):
        images = images.cuda()

        loss, recons = distr_vae(
            images,
            return_loss = True,
            return_recons = True,
            temp = temp
        )

        if args.deepspeed:
            # Gradients are automatically zeroed after the step
            distr_vae.backward(loss)
            distr_vae.step()
        else:
            opt.zero_grad()
            loss.backward()
            opt.step()

        logs = {}

        if i % 100 == 0:
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

            save_model(f'./vae.pt')
            wandb.save('./vae.pt')

            # temperature anneal

            temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

            # lr decay

            sched.step()

        if i % 10 == 0:
            lr = sched.get_last_lr()[0]
            print(epoch, i, f'lr - {lr:6f} loss - {loss.item()}')

            logs = {
                **logs,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item(),
                'lr': lr
            }

        wandb.log(logs)
        global_step += 1

    # save trained model to wandb as an artifact every epoch's end

    model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
    model_artifact.add_file('vae.pt')
    run.log_artifact(model_artifact)

# save final vae and cleanup

save_model('./vae-final.pt')
wandb.save('./vae-final.pt')

model_artifact = wandb.Artifact('trained-vae', type = 'model', metadata = dict(model_config))
model_artifact.add_file('vae-final.pt')
run.log_artifact(model_artifact)

wandb.finish()
