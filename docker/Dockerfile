
ARG IMG_TAG=1.6.0-cuda10.1-cudnn7-devel
ARG IMG_REPO=pytorch
ARG BRANCH=main
ARG REMOTE=lucidrains

FROM pytorch/$IMG_REPO:$IMG_TAG

RUN apt-get -y update && apt-get -y install git gcc llvm-9-dev cmake libaio-dev

RUN git clone https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed
RUN cd /tmp/DeepSpeed && DS_BUILD_SPARSE_ATTN=1 ./install.sh -r
RUN pip install git+https://github.com/janEbert/DALLE-pytorch.git@ds-checkpointing

WORKDIR dalle
