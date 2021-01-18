## DALL-E in Pytorch

Implementation / replication of <a href="https://openai.com/blog/dall-e/">DALL-E</a>, OpenAI's Text to Image Transformer, in Pytorch. It will also contain <a href="https://openai.com/blog/clip/">CLIP</a> for ranking the generations.

<a href="https://github.com/sdtblck">Sid</a>, <a href="http://github.com/kingoflolz">Ben</a>, and <a href="https://github.com/AranKomat">Aran</a> over at <a href="https://www.eleuther.ai/">Eleuther AI</a> are working on <a href="https://github.com/EleutherAI/DALLE-mtf">DALL-E for Mesh Tensorflow</a>! Please lend them a hand if you would like to see DALL-E trained on TPUs.

<a href="https://www.youtube.com/watch?v=j4xgkjWlfL4">Yannic Kilcher's video</a>

Before we replicate this, we can settle for <a href="https://github.com/lucidrains/deep-daze">Deep Daze</a> or <a href="https://github.com/lucidrains/big-sleep">Big Sleep</a>

## Status

<a href="https://github.com/htoyryla">Hannu</a> has managed to train a small 6 layer DALL-E on a dataset of just 2000 landscape images! (2048 visual tokens)

<img src="./images/landscape.png"></img>

## Install

```bash
$ pip install dalle-pytorch
```

## Usage

Train VAE

```python
import torch
from dalle_pytorch import DiscreteVAE

vae = DiscreteVAE(
    image_size = 256,
    num_layers = 3,          # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = 8192,       # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = 512,      # codebook dimension
    hidden_dim = 64,         # hidden dimension
    num_resnet_blocks = 1,   # number of resnet blocks
    temperature = 0.9,       # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = False # straight-through for gumbel softmax. unclear if it is better one way or the other
)

images = torch.randn(4, 3, 256, 256)

loss = vae(images, return_recon_loss = True)
loss.backward()

# train with a lot of data to learn a good codebook
```

Train DALL-E with pretrained VAE from above

```python
import torch
from dalle_pytorch import DiscreteVAE, DALLE

vae = DiscreteVAE(
    image_size = 256,
    num_layers = 3,
    num_tokens = 8192,
    codebook_dim = 1024,
    hidden_dim = 64,
    num_resnet_blocks = 1,
    temperature = 0.9
)

dalle = DALLE(
    dim = 1024,
    vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = 10000,    # vocab size for text
    text_seq_len = 256,         # text sequence length
    depth = 12,                 # should aim to be 64
    heads = 16,                 # attention heads
    dim_head = 64,              # attention head dimension
    attn_dropout = 0.1,         # attention dropout
    ff_dropout = 0.1            # feedforward dropout
)

text = torch.randint(0, 10000, (4, 256))
images = torch.randn(4, 3, 256, 256)
mask = torch.ones_like(text).bool()

loss = dalle(text, images, mask = mask, return_loss = True)
loss.backward()

# do the above for a long time with a lot of data ... then

images = dalle.generate_images(text, mask = mask)
images.shape # (2, 3, 256, 256)
```

## Ranking the generations

Train CLIP

```python
import torch
from dalle_pytorch import CLIP

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 10000,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    num_visual_tokens = 512,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
)

text = torch.randint(0, 10000, (4, 256))
images = torch.randn(4, 3, 256, 256)
mask = torch.ones_like(text).bool()

loss = clip(text, images, text_mask = mask, return_loss = True)
loss.backward()
```

To get the similarity scores from your trained Clipper, just do

```python
images, scores = dalle.generate_images(text, mask = mask, clip = clip)

scores.shape # (2,)
images.shape # (2, 3, 256, 256)

# do your topk here, in paper they sampled 512 and chose top 32
```

Or you can just use the official <a href="https://github.com/openai/CLIP">CLIP model</a> to rank the images from DALL-E

## Scaling depth

In the blog post, they used 64 layers to achieve their results. I added reversible networks, from the <a href="https://github.com/lucidrains/reformer-pytorch">Reformer</a> paper, in order for users to attempt to scale depth at the cost of compute. Reversible networks allow you to scale to any depth at no memory cost, but a little over 2x compute cost (each layer is rerun on the backward pass).

Simply set the `reversible` keyword to `True` for the `DALLE` class

```python
dalle = DALLE(
    dim = 1024,
    vae = vae,
    num_text_tokens = 10000,
    text_seq_len = 256,
    depth = 64,
    heads = 16,
    reversible = True  # <-- reversible networks https://arxiv.org/abs/2001.04451
)
```

## Sparse Attention

You can also train with Microsoft Deepspeed's <a href="https://www.deepspeed.ai/news/2020/09/08/sparse-attention.html">Sparse Attention</a>, with any combination of dense and sparse attention that you'd like. However, you will have to endure the installation process.

First, you need to install Deepspeed with Sparse Attention

```bash
$ sh install_deepspeed.sh
```

Next, you need to install the pip package `triton`

```bash
$ pip install triton
```

If both of the above succeeded, now you can train with Sparse Attention!

```python
dalle = DALLE(
    dim = 512,
    vae = vae,
    num_text_tokens = 10000,
    text_seq_len = 256,
    depth = 64,
    heads = 8,
    sparse_attn = (True, False) * 32  # interleave sparse and dense attention for 64 layers
)
```

## Citations

```bibtex
@misc{unpublished2021dalle,
    title   = {DALL·E: Creating Images from Text},
    author  = {Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray},
    year    = {2021}
}
```

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{kitaev2020reformer,
    title   = {Reformer: The Efficient Transformer},
    author  = {Nikita Kitaev and Łukasz Kaiser and Anselm Levskaya},
    year    = {2020},
    eprint  = {2001.04451},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

*Those who do not want to imitate anything, produce nothing.* - Dali
