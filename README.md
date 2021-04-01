# `perugia`

> "Those who do not want to imitate anything, produce nothing."
>  \- Dali

`perugia` is a text-to-image generator. 

Give it a phrase - it tries to recreate an image. 

If you're looking for a version of this that, you know, actually works presently, you should check out:
- [deep-daze](https://github.com/lucidrains/deep-daze)
- [big-sleep](https://github.com/lucidrains/big-sleep)

For now - there's no pretrained model. You can attempt to train one yourself however -

- [Getting Started](https://github.com/afiaka87/perugia/wiki/Getting-Started)

- [![Original DALLE-pytorch Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dWvA54k4fH8zAmiix3VXbg95uEIMfqQM?usp=sharing)  Original DALLE-pytorch Notebook

## Chat

- This project lives on app.element.io:
https://app.element.io/#/group/+afiaka87-perugia:matrix.org
  

## Horovod (nvidia-docker image)
```
nvidia-docker pull horovod:latest
nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest
```

## Install NVCCL
```
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
apt update
apt install libnccl2=2.7.6-1+cuda10.1 libnccl-dev=2.7.6-1+cuda10.1

```

## Horovod Support
- 
```bash
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip --no-cache-dir install horovod[pytorch]

```

```
horovodrun --check-build
```
You should see the following output if the build was successful:
```
Horovod v0.21.3:

Available Frameworks:
    [ ] TensorFlow
    [X] PyTorch
    [ ] MXNet

Available Controllers:
    [ ] MPI
    [X] Gloo

Available Tensor Operations:
    [ ] NCCL
    [ ] DDL
    [ ] CCL
    [ ] MPI
    [X] Gloo
```

*Note: Some models with a high computation to communication ratio benefit from doing allreduce on CPU, even if a GPU version is available. To force allreduce to happen on CPU, pass device_dense='/cpu:0' to hvd.DistributedOptimizer:*
```python
opt = hvd.DistributedOptimizer(opt, device_dense='/cpu:0')
```


## Disclaimer
  - This is a fork of the repository https://github.com/lucidrains/DALLE-pytorch designed to be a bit easier to use.

  - This isn't he same DALL-E that OpenAI have presented. It is an attempt at recreating its architecture based on the details released by those researchers. There's not a pretrained model yet, but I believe one is right around the corner.
  
  - I have not been able to test this code out on Windows 10 yet. I hope that it works using the conda instructions, but please file an issue if it doesn't. 


## Citations

https://github.com/afiaka87/perugia/wiki/Citations