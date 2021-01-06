## DALL-E in Pytorch (wip)

Implementation / replication of <a href="https://openai.com/blog/dall-e/">DALL-E</a>, OpenAI's Text to Image Transformer, in Pytorch. It will also contain <a href="https://openai.com/blog/clip/">CLIP</a> for ranking the generations.

<a href="https://www.youtube.com/watch?v=j4xgkjWlfL4">Yannic Kilcher's video</a>

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
    num_tokens = 2000,
    dim = 512,
    hidden_dim = 64
)

images = torch.randn(8, 3, 256, 256)

loss = vae(images, return_recon_loss = True)
loss.backward()
```

Train CLIP

```python
import torch
from dalle_pytorch import CLIP

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 10000,
    num_visual_tokens = 512,
    text_enc_depth = 6,
    visual_enc_depth = 6,
    text_seq_len = 256,
    visual_seq_len = 1024,
    text_heads = 8,
    visual_heads = 8
)

text = torch.randint(0, 10000, (2, 256))
images = torch.randint(0, 512, (2, 1024))
mask = torch.ones_like(text).bool()

loss = clip(text, images, text_mask = mask, return_loss = True)
loss.backward()
```

Train DALL-E

```python
import torch
from dalle_pytorch import DALLE

dalle = DALLE(
    dim = 512,
    num_text_tokens = 10000,
    num_image_tokens = 512,
    text_seq_len = 256,
    image_seq_len = 1024,
    depth = 6, # should be 64
    heads = 8
)

text = torch.randint(0, 10000, (2, 256))
images = torch.randint(0, 512, (2, 1024))
mask = torch.ones_like(text).bool()

loss = dalle(text, images, mask = mask, return_loss = True)
loss.backward()
```

Combine pretrained VAE with DALL-E, and pass in raw images

```python
import torch
from dalle_pytorch import DiscreteVAE, DALLE

vae = DiscreteVAE(
    num_tokens = 512,
    dim = 512
)

dalle = DALLE(
    dim = 512,
    vae = vae,
    num_text_tokens = 10000,
    num_image_tokens = 512,
    text_seq_len = 256,
    image_seq_len = 1024,
    depth = 6, # should be 64
    heads = 8
)

text = torch.randint(0, 10000, (2, 256))
images = torch.randn(2, 3, 256, 256) # train directly on raw images, VAE converts to proper embeddings
mask = torch.ones_like(text).bool()

loss = dalle(text, images, mask = mask, return_loss = True)
loss.backward()
```

Finally, to generate images

```python
from dalle_pytorch import generate_images

images = generate_images(
    dalle,
    vae = vae,
    text = text,
    mask = mask
)

images.shape # (2, 3, 256, 256)
```

To get the similarity scores from your trained Clipper, just do

```python
from dalle_pytorch import generate_images

images, scores = generate_images(
    dalle,
    vae = vae,
    text = text,
    mask = mask,
    clipper = clip
)

scores.shape # (2,)
images.shape # (2, 3, 256, 256)

# do your topk here, in paper they sampled 512 and chose top 32
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
