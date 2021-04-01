This is a fork of the repository https://github.com/lucidrains/DALLE-pytorch designed to be a bit easier to use.

"Those who do not want to imitate anything, produce nothing." - Dali

**disclaimer**: There is often confusion surrounding this. This is _not_ the same DALL-E that OpenAI have presented. It is an attempt at recreating its architecture based on the details released by those researchers. There's not a pretrained model yet, but I believe one is right around the corner.

# Text to Image 

## Dependencies
- llvm-9-dev
- cmake
- gcc
- python3.7.x

## Installation

https://github.com/afiaka87/text_to_image/wiki/Installation

## Training From Scratch
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dWvA54k4fH8zAmiix3VXbg95uEIMfqQM?usp=sharing) Train in Colab
- https://github.com/afiaka87/text_to_image/wiki/Train-with-your-Own-Dataset

## Generate images from text
(WIP)

## VRAM Optimizations:
https://github.com/afiaka87/text_to_image/wiki/Memory-Speed-Optimizations


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
