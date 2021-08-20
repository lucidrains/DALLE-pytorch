# DALL-E in Pytorch

<p align='center'>
  <a href="https://colab.research.google.com/gist/afiaka87/b29213684a1dd633df20cab49d05209d/train_dalle_pytorch.ipynb">
         <img alt="Train DALL-E w/ DeepSpeed" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
  <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a></br>
  <a href="https://github.com/robvanvolt/DALLE-models">Released DALLE Models</a></br>
  <a href="https://github.com/rom1504/dalle-service">Web-Hostable DALLE Checkpoints</a></br>

  <a href="https://www.youtube.com/watch?v=j4xgkjWlfL4">Yannic Kilcher's video</a>
<p>
Implementation / replication of <a href="https://openai.com/blog/dall-e/">DALL-E</a> (<a href="https://arxiv.org/abs/2102.12092">paper</a>), OpenAI's Text to Image Transformer, in Pytorch.  It will also contain <a href="https://openai.com/blog/clip/">CLIP</a> for ranking the generations.

---



[Quick Start](https://github.com/lucidrains/DALLE-pytorch/wiki)

<a href="https://github.com/lucidrains/deep-daze">Deep Daze</a> or <a href="https://github.com/lucidrains/big-sleep">Big Sleep</a> are great alternatives!

## Appreciation
  
This library could not have been possible without the contributions of <a href="https://github.com/janEbert">janEbert</a>, <a href="https://github.com/afiaka87">Clay</a>, <a href="https://github.com/robvanvolt">robvanvolt</a>, and <a href="https://github.com/rom1504">Romaine</a>! 🙏

## Status
<p align='center'>

- <a href="https://github.com/htoyryla">Hannu</a> has managed to train a small 6 layer DALL-E on a dataset of just 2000 landscape images! (2048 visual tokens)

<img src="./images/landscape.png"></img>

- <a href="https://github.com/kobiso">Kobiso</a>, a research engineer from Naver, has trained on the CUB200 dataset <a href="https://github.com/lucidrains/DALLE-pytorch/discussions/131">here</a>, using full and deepspeed sparse attention

<img src="./images/birds.png" width="256"></img>

- (3/15/21) <a href="https://github.com/afiaka87">afiaka87</a> has managed one epoch using a reversible DALL-E and the dVaE <a href="https://github.com/lucidrains/DALLE-pytorch/issues/86#issue-832121328">here</a>

- <a href="https://github.com/robvanvolt">TheodoreGalanos</a> has trained on 150k layouts with the following results
<p>
  <img src="./images/layouts-1.jpg" width="256"></img>
  <img src="./images/layouts-2.jpg" width="256"></img>
</p>
- <a href="https://github.com/rom1504">Rom1504</a> has trained on 50k fashion images with captions with a really small DALL-E (2 layers) for just 24 hours with the following results
<p/>
<img src="./images/clothing.png" width="420"></img>

- <a href="https://github.com/afiaka87">afiaka87</a> trained for 6 epochs on the same dataset as before thanks to the efficient 16k VQGAN with the following <a href="https://github.com/lucidrains/DALLE-pytorch/discussions/322>discussion">results</a>

<p align='centered'>
  <img src="https://user-images.githubusercontent.com/3994972/123564891-b6f18780-d780-11eb-9019-8a1b6178f861.png" width="420" alt-text='a photo of westwood park, san francisco, from the water in the afternoon'></img>
  <img src="https://user-images.githubusercontent.com/3994972/123564776-4c404c00-d780-11eb-9c8e-3356df358df3.png" width="420" alt-text='a female mannequin dressed in an olive button-down shirt and gold palazzo pants'> </img>
</p>
  
Thanks to the amazing "mega b#6696" you can generate from this checkpoint in colab - 
<a href="https://colab.research.google.com/drive/11V2xw1eLPfZvzW8UQyTUhqCEU71w6Pr4?usp=sharing">
  <img alt="Run inference on the Afiaka checkpoint in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>
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
    num_layers = 3,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = 8192,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = 512,       # codebook dimension
    hidden_dim = 64,          # hidden dimension
    num_resnet_blocks = 1,    # number of resnet blocks
    temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
)

images = torch.randn(4, 3, 256, 256)

loss = vae(images, return_loss = True)
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
images.shape # (4, 3, 256, 256)
```

To prime with a starting crop of an image, simply pass two more arguments

```python
img_prime = torch.randn(4, 3, 256, 256)

images = dalle.generate_images(
    text,
    mask = mask,
    img = img_prime,
    num_init_img_tokens = (14 * 32)  # you can set the size of the initial crop, defaults to a little less than ~1/2 of the tokens, as done in the paper
)

images.shape # (4, 3, 256, 256)
```

You may also want to generate text using DALL-E. For that call this function:

```python
text_tokens, texts = dalle.generate_texts(tokenizer, text)
```

## OpenAI's Pretrained VAE

You can also skip the training of the VAE altogether, using the pretrained model released by OpenAI! The wrapper class should take care of downloading and caching the model for you auto-magically.

```python
import torch
from dalle_pytorch import OpenAIDiscreteVAE, DALLE

vae = OpenAIDiscreteVAE()       # loads pretrained OpenAI VAE

dalle = DALLE(
    dim = 1024,
    vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = 10000,    # vocab size for text
    text_seq_len = 256,         # text sequence length
    depth = 1,                  # should aim to be 64
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
```

## Taming Transformer's Pretrained VQGAN VAE

You can also use the pretrained VAE offered by the authors of <a href="https://github.com/CompVis/taming-transformers">Taming Transformers</a>! Currently only the VAE with a codebook size of 1024 is offered, with the hope that it may train a little faster than OpenAI's, which has a size of 8192.

In contrast to OpenAI's VAE, it also has an extra layer of downsampling, so the image sequence length is 256 instead of 1024 (this will lead to a 16 reduction in training costs, when you do the math). Whether it will generalize as well as the original DALL-E is up to the citizen scientists out there to discover.

Update - <a href="https://github.com/lucidrains/DALLE-pytorch/discussions/131">it works!</a>

```python
from dalle_pytorch import VQGanVAE

vae = VQGanVAE()

# the rest is the same as the above example
```

The default VQGan is the codebook size 1024 one trained on imagenet. If you wish to use a different one, you can use the `vqgan_model_path` and `vqgan_config_path` to pass the .ckpt file and the .yaml file. These options can be used both in train-dalle script or as argument of VQGanVAE class. Other pretrained VQGAN can be found in [taming transformers readme](https://github.com/CompVis/taming-transformers#overview-of-pretrained-models). If you want to train a custom one you can [follow this guide](https://github.com/CompVis/taming-transformers/pull/54)

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

The blogpost alluded to a mixture of different types of sparse attention, used mainly on the image (while the text presumably had full causal attention). I have done my best to replicate these types of sparse attention, on the scant details released. Primarily, it seems as though they are doing causal axial row / column attention, combined with a causal convolution-like attention.

By default `DALLE` will use full attention for all layers, but you can specify the attention type per layer as follows.

- `full` full attention

- `axial_row` axial attention, along the rows of the image feature map

- `axial_col` axial attention, along the columns of the image feature map

- `conv_like` convolution-like attention, for the image feature map

The sparse attention only applies to the image. Text will always receive full attention, as said in the blogpost.

```python
dalle = DALLE(
    dim = 1024,
    vae = vae,
    num_text_tokens = 10000,
    text_seq_len = 256,
    depth = 64,
    heads = 16,
    reversible = True,
    attn_types = ('full', 'axial_row', 'axial_col', 'conv_like')  # cycles between these four types of attention
)
```

## Deepspeed Sparse Attention

You can also train with Microsoft Deepspeed's <a href="https://www.deepspeed.ai/news/2020/09/08/sparse-attention.html">Sparse Attention</a>, with any combination of dense and sparse attention that you'd like. However, you will have to endure the installation process.

First, you need to install Deepspeed with Sparse Attention

```bash
$ sh install_deepspeed.sh
```

Next, you need to install the pip package `triton`. It will need to be a version `< 1.0` because that's what Microsoft used.

```bash
$ pip install triton==0.4.2
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
    attn_types = ('full', 'sparse')  # interleave sparse and dense attention for 64 layers
)
```

## Training

This section will outline how to train the discrete variational autoencoder as well as the final multi-modal transformer (DALL-E). We are going to use <a href="https://wandb.ai/">Weights & Biases</a> for all the experiment tracking.

(You can also do everything in this section in a Google Colab, link below)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dWvA54k4fH8zAmiix3VXbg95uEIMfqQM?usp=sharing) Train in Colab

```bash
$ pip install wandb
```

Followed by

```bash
$ wandb login
```

### VAE

To train the VAE, you just need to run

```python
$ python train_vae.py --image_folder /path/to/your/images
```

If you installed everything correctly, a link to the experiments page should show up in your terminal. You can follow your link there and customize your experiment, like the example layout below.

<img src="./images/wb.png" width="700px"></img>

You can of course open up the training script at `./train_vae.py`, where you can modify the constants, what is passed to Weights & Biases, or any other tricks you know to make the VAE learn better.

Model will be saved periodically to `./vae.pt`

In the experiment tracker, you will have to monitor the hard reconstruction, as we are essentially teaching the network to compress images into discrete visual tokens for use in the transformer as a visual vocabulary.

Weights and Biases will allow you to monitor the temperature annealing, image reconstructions (encoder and decoder working properly), as well as to watch out for codebook collapse (where the network decides to only use a few tokens out of what you provide it).

Once you have trained a decent VAE to your satisfaction, you can move on to the next step with your model weights at `./vae.pt`.

### DALL-E Training

## Training using an Image-Text-Folder

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

## Training using WebDataset

WebDataset files are regular .tar(.gz) files which can be streamed and used for DALLE-pytorch training.
You Just need to provide the image (first comma separated argument) and caption (second comma separated argument) 
column key after the --wds argument. The ---image_text_folder points to your .tar(.gz) file instead of the datafolder.

```python
$ python train_dalle.py --wds img,cap --image_text_folder /path/to/data.tar(.gz)
```

Distributed training with deepspeed works the same way, e.g.:

```python
$ deepspeed train_dalle.py --wds img,cap --image_text_folder /path/to/data.tar(.gz) --fp16 --deepspeed
```

If you have containing shards (dataset split into several .tar(.gz) files), this is also supported:

```python
$ deepspeed train_dalle.py --wds img,cap --image_text_folder /path/to/shardfolder --fp16 --deepspeed
```

You can stream the data from a http server or gloogle cloud storage like this:

```python
$ deepspeed train_dalle.py --image_text_folder "http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar" --wds jpg,json --taming --truncate_captions --random_resize_crop_lower_ratio=0.8 --attn_types=full --epochs=2 --fp16 --deepspeed
```

In order to convert your image-text-folder to WebDataset format, you can make use of one of several methods.
(https://www.youtube.com/watch?v=v_PacO-3OGQ here are given 4 examples, or a little helper script which also supports splitting your dataset
into shards of .tar.gz files https://github.com/robvanvolt/DALLE-datasets/blob/main/wds_create_shards.py)

### DALL-E with OpenAI's VAE

You can now also train DALL-E without having to train the Discrete VAE at all, courtesy to their open-sourcing their model. You simply have to invoke the `train_dalle.py` script without specifying the `--vae_path`

```python
$ python train_dalle.py --image_text_folder /path/to/coco/dataset
```

### DALL-E with Taming Transformer's VQVAE

Just use the `--taming` flag. Highly recommended you use this VAE over the OpenAI one!

```python
$ python train_dalle.py --image_text_folder /path/to/coco/dataset --taming
```

### Generation

Once you have successfully trained DALL-E, you can then use the saved model for generation!

```python
$ python generate.py --dalle_path ./dalle.pt --text 'fireflies in a field under a full moon'
```

You should see your images saved as `./outputs/{your prompt}/{image number}.jpg`

To generate multiple images, just pass in your text with '|' character as a separator.

ex.

```python
$ python generate.py --dalle_path ./dalle.pt --text 'a dog chewing a bone|a cat chasing mice|a frog eating a fly'
```

Note that DALL-E is a full image+text language model. As a consequence you can also generate text using a dalle model.

```python
$ python generate.py --dalle_path ./dalle.pt --text 'a dog chewing a bone' --gentext
```

This will complete the provided text, save it in a caption.txt and generate the corresponding images.

### Docker

You can use a docker container to make sure the version of Pytorch and Cuda are correct for training DALL-E. <a href="https://docs.docker.com/get-docker/">Docker</a> and <a href='#'>Docker Container Runtime</a> should be installed.

To build:

```bash
docker build -t dalle docker
```

To run in an interactive shell:

```bash
docker run --gpus all -it --mount src="$(pwd)",target=/workspace/dalle,type=bind dalle:latest bash
```

### Distributed Training

#### DeepSpeed

Thanks to <a href="https://github.com/janEbert">janEbert</a>, the repository is now equipped so you can train DALL-E with Microsoft's <a href="https://www.deepspeed.ai/">Deepspeed</a>!

You can simply replace any `$ python <file>.py [args...]` command with

```sh
$ deepspeed <file>.py [args...] --deepspeed
```

to use the aforementioned DeepSpeed library for distributed training, speeding up your experiments.

Modify the `deepspeed_config` dictionary in `train_dalle.py` or
`train_vae.py` according to the DeepSpeed settings you'd like to use
for each one. See the [DeepSpeed configuration
docs](https://www.deepspeed.ai/docs/config-json/) for more
information.

#### DeepSpeed - 32 and 16 bit Precision
As of DeepSpeed version 0.3.16, ZeRO optimizations can be used with
single-precision floating point numbers. If you are using an older
version, you'll have to pass the `--fp16` flag to be able to enable
ZeRO optimizations.


#### DeepSpeed - Apex Automatic Mixed Precision.
Automatic mixed precision is a stable alternative to fp16 which still provides a decent speedup.
In order to run with Apex AMP (through DeepSpeed), you will need to install DeepSpeed using either the Dockerfile or the bash script.

Then you will need to install apex from source. 
This may take awhile and you may see some compilation warnings which can be ignored. 
```sh
sh install_apex.sh
```

Now, run `train_dalle.py` with `deepspeed` instead of `python` as done here:
```sh
deepspeed train_dalle.py \
    --taming \
    --image_text_folder 'DatasetsDir' \
    --distr_backend 'deepspeed' \
    --amp
```

#### Horovod

[Horovod](https://horovod.ai) offers a stable way for data parallel
training.

After [installing
Horovod](https://github.com/lucidrains/DALLE-pytorch/wiki/Horovod-Installation),
replace any `$ python <file>.py [args...]` command with

```sh
$ horovodrun -np <num-gpus> <file>.py [args...] --distributed_backend horovod
```

to use the Horovod library for distributed training, speeding up your
experiments. This will multiply your effective batch size per training
step by `<num-gpus>`, so you may need to rescale the learning rate
accordingly.

#### Custom Tokenizer

This repository supports custom tokenization with <a href="https://github.com/VKCOM/YouTokenToMe">YouTokenToMe</a>, if you wish to use it instead of the default simple tokenizer. Simply pass in an extra `--bpe_path` when invoking `train_dalle.py` and `generate.py`, with the path to your BPE model file.

The only requirement is that you use `0` as the padding during tokenization

ex.

```sh
$ python train_dalle.py --image_text_folder ./path/to/data --bpe_path ./path/to/bpe.model
```

To create a BPE model file from scratch, firstly

```bash
$ pip install youtokentome
```

Then you need to prepare a big text file that is a representative sample of the type of text you want to encode. You can then invoke the `youtokentome` command-line tools. You'll also need to specify the vocab size you wish to use, in addition to the corpus of text.

```bash
$ yttm bpe --vocab_size 8000 --data ./path/to/big/text/file.txt --model ./path/to/bpe.model
```

That's it! The BPE model file is now saved to `./path/to/bpe.model` and you can begin training!

#### Chinese

You can train with a <a href="https://huggingface.co/bert-base-chinese">pretrained chinese tokenizer</a> offered by Huggingface 🤗 by simply passing in an extra flag `--chinese`

ex.

```sh
$ python train_dalle.py --chinese --image_text_folder ./path/to/data
```

```sh
$ python generate.py --chinese --text '追老鼠的猫'
```

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

```bibtex
@misc{ding2021cogview,
    title   = {CogView: Mastering Text-to-Image Generation via Transformers},
    author  = {Ming Ding and Zhuoyi Yang and Wenyi Hong and Wendi Zheng and Chang Zhou and Da Yin and Junyang Lin and Xu Zou and Zhou Shao and Hongxia Yang and Jie Tang},
    year    = {2021},
    eprint  = {2105.13290},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@software{peng_bo_2021_5196578,
    author       = {PENG Bo},
    title        = {BlinkDL/RWKV-LM: 0.01},
    month        = {aug},
    year         = {2021},
    publisher    = {Zenodo},
    version      = {0.01},
    doi          = {10.5281/zenodo.5196578},
    url          = {https://doi.org/10.5281/zenodo.5196578}
}
```

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

*Those who do not want to imitate anything, produce nothing.* - Dali
