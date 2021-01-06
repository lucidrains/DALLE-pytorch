## DALL-E Pytorch (wip)

Implementation / replication of <a href="https://openai.com/blog/dall-e/">DALL-E</a>, OpenAI's Text to Image Transformer, in Pytorch. It will also contain <a href="https://openai.com/blog/clip/">CLIP</a> for ranking the generations.

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
	num_tokens = 2000
)

x = torch.randn(8, 3, 256, 256)
loss = vae(x, return_recon_loss)
loss.backward()
```

Train CLIP

```python
import torch
from dalle_pytorch import CLIP

clip = CLIP(
	dim = 512
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
	dim = 512
)

text = torch.randint(0, 10000, (2, 256))
images = torch.randint(0, 512, (2, 1024))
mask = torch.ones_like(text).bool()

loss = dalle(text, images, mask = mask, return_loss = True)
loss.backward()
```

## Citations

```bibtex
@misc{unpublished2021dalle,
    title   = {DALL·E: Creating Images from Text},
    author  = {Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray},
    year    = {2021}
}
```
