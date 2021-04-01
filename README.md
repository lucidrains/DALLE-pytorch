This is a fork of the repository https://github.com/lucidrains/DALLE-pytorch designed to be a bit easier to use.

**disclaimer**: There is often confusion surrounding this. This is _not_ the same DALL-E that OpenAI have presented. It is an attempt at recreating its architecture based on the details released by those researchers. There's not a pretrained model yet, but I believe one is right around the corner.

# Text to Image 

## Dependencies
- llvm-9-dev
- cmake
- gcc
- python3.7.x

## Installation

https://github.com/afiaka87/text_to_image/wiki/Installation

## Usage

### Training
https://github.com/afiaka87/text_to_image/wiki/Train-with-your-Own-Dataset

### Generate images from text


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

Begin training:

```sh
# Leave off the `-taming` parameter to use OpenAI's VAE
python train_dalle.py --image_text_folder /path/to/data -taming
```

You can stop anytime you'd like and resume later:
```sh
python train_dalle.py --image_text_folder /path/to/data -taming --dalle_path ./dalle.pt 
```

### Generate Images from Text 

Finally - the cool part.

** You will need a pretrained `dalle.pt` checkpoint in order to run this. You may attempt to train one yourself but so far there are none worth releasing. **

```python
$ python generate.py --dalle_path ./dalle.pt --text 'fireflies in a field under a full moon'
```

You should see your images saved as `./outputs/{your prompt}/{image number}.jpg`

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
