This is a fork of the repository https://github.com/lucidrains/DALLE-pytorch containing additional features and bug fixes. Please go to that repo for status updates on training efforts and probably other features as well, although I will attempt to merge their changes downstream as often as possible.

## DALL-E in Pytorch
    num_text_tokens = 10000,    # vocab size for text
    text_seq_len = 256,         # text sequence length
    depth = 1,                  # should aim to be 64
    heads = 16,                 # attention heads
    dim_head = 64,              # attention head dimension
    attn_dropout = 0.1,         # attention dropout
    ff_dropout = 0.1            # feedforward dropout
## Dependencies
- llvm-9-dev
- cmake
- gcc
- python3.7.x

## Installation instructions for Ubuntu 20.04 (Python3.7 is required)

### First - install dependencies
```sh
sudo apt-get -y install llvm-9-dev cmake
git clone https://github.com/microsoft/DeepSpeed.git /tmp/Deepspeed
cd /tmp/Deepspeed && DS_BUILD_SPARSE_ATTN=1 ./install.sh -s
pip install triton
cd ~
```

### Conda
```bash
#!/bin/bash

conda create -n dalle_pytorch_afiaka87 python=3.7
conda activate dalle_pytorch_afiaka87
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install "git+https://github.com:afiaka87/DALLE-pytorch.git"
```

### Pip
```bash
#!/bin/bash

python -m pip install virtualenv
python -m virtualenv -p=python3.7 ~/.virtualenvs/dalle_pytorch_afiaka87
source ~/.virtualenvs/dalle_pytorch_afiaka87/bin/activate
# Make sure your terminal shows that you're inside the virtual environment - and then run:
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com:afiaka87/DALLE-pytorch.git"
```

## Usage

There are two pretrained VAE's to choose from. You may also train one yourself - you can find instructions on lucidrains repo. 

### taming.VQGanVAE1024

#### Low VRAM/Fairly Accurate

Offered generously by the authors of <a href="https://github.com/CompVis/taming-transformers">Taming Transformers</a>
This VAE is capable of generalizing and is also quite a bit easier to run than OpenAI's pretrained VAE (see below)
A theoretical speedup of 16x is possible - although it seems to be more in the range of 2-4x in my experience. 
VRAM savings are substantial.

```python
from dalle_pytorch import VQGanVAE1024
vae = VQGanVAE1024()
```

### dalle_pytorch.OpenAIDiscreteVAE
#### (Accurate - More VRAM)
```python
from dalle_pytorch import OpenAIDiscreteVAE, DALLE
vae = OpenAIDiscreteVAE()
```

### Instantiate DALLE with your VAE of choice
dalle = DALLE(
    dim = 1024,
    vae = vae,
)

###
```python
text = torch.randint(0, 10000, (4, 256))
images = torch.randn(4, 3, 256, 256)
mask = torch.ones_like(text).bool()

loss = dalle(text, images, mask = mask, return_loss = True)
loss.backward()
```

## Ranking the generations

You can use the official <a href="https://github.com/openai/CLIP">CLIP model</a> to rank the images from DALL-E. 

## VRAM Optimizations:

### Reversible
Simply set the `reversible` keyword to `True` for the `DALLE` class

```python
dalle = DALLE(
    # ...
    reversible = True  # <-- reversible networks https://arxiv.org/abs/2001.04451
)
```

### Sparse Attention (MS deepspeed)

```python
dalle = DALLE(
    # ...
    attn_types = ('sparse')  # cycles between these four types of attention
)


### Other attetion layers:

By default `DALLE` will use full attention for all layers, but you can specify the attention type per layer as follows.

- `full` full attention
- `axial_row` axial attention, along the rows of the image feature map
- `axial_col` axial attention, along the columns of the image feature map
- `conv_like` convolution-like attention, for the image feature map

```python
dalle = DALLE(
    # ...
    attn_types = ('full', 'axial_row', 'axial_col', 'conv_like')  # cycles between these four types of attention
)
```

## Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dWvA54k4fH8zAmiix3VXbg95uEIMfqQM?usp=sharing) Train in Colab

Assuming you have installed all dependencies -
```bash
# Make sure you're in a virtual environment on either conda or python-virtualenv:
# conda activate dalle_pytorch_afiaka87
# source ~/.virtualenvs/dalle_pytorch_afiaka87
```

Now you just have to invoke the `./train_dalle.py` script, indicating which VAE model you would like to use, as well as the path to your folder if images and text.

The dataset I am currently working with contains a folder of images and text files, arbitraily nested in subfolders, where text file name corresponds with the image name, and where each text file contains multiple descriptions, delimited by newlines. The script will find and pair all the image and text files with the same names, and randomly select one of the textual descriptions during batch creation.

ex.

```
📂image-and-text-data
 ┣ 📜cat.png
 ┣ 📜cat.txt
 ┣ 📜dog.jpg
 ┣ 📜dog.txt
 ┣ 📜turtle.jpeg
 ┗ 📜turtle.txt
```

ex. `cat.txt`

```text
A black and white cat curled up next to the fireplace
A fireplace, with a cat sleeping next to it
A black cat with a red collar napping
```

If you have a dataset with its own directory structure for tying together image and text descriptions, do let me know in the issues, and I'll see if I can accommodate it in the script.

```python
$ python train_dalle.py --vae_path ./vae.pt --image_text_folder /path/to/data
```

You likely will not finish DALL-E training as quickly as you did your Discrete VAE. To resume from where you left off, just run the same script, but with the path to your DALL-E checkpoints.

```python
$ python train_dalle.py --dalle_path ./dalle.pt --image_text_folder /path/to/data
```


### Generate Images from Text 

** You will need a pretrained `dalle.pt` checkpoint in order to run this. You may attempt to train one yourself but so far there are none worth releasing. **

```python
$ python generate.py --dalle_path ./dalle.pt --text 'fireflies in a field under a full moon'
```

You should see your images saved as `./outputs/{your prompt}/{image number}.jpg`

### Distributed Training with DeepSpeed

You can replace any `$ python <file>.py [args...]` command with
```sh
$ deepspeed <file>.py [args...] --deepspeed
```
to use the aforementioned DeepSpeed library for distributed training, speeding up your experiments.

## Citations

```bibtex
@misc{ramesh2021zeroshot,
    title   = {Zero-Shot Text-to-Image Generation}, 
    author  = {Aditya Ramesh and Mikhail Pavlov and Gabriel Goh and Scott Gray and Chelsea Voss and Alec Radford and Mark Chen and Ilya Sutskever},
    year    = {2021},
    eprint  = {2102.12092},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
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

```bibtex
@misc{esser2021taming,
    title   = {Taming Transformers for High-Resolution Image Synthesis},
    author  = {Patrick Esser and Robin Rombach and Björn Ommer},
    year    = {2021},
    eprint  = {2012.09841},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

*Those who do not want to imitate anything, produce nothing.* - Dali
