
ARG IMG_TAG=1.8.1-cuda10.2-cudnn7-devel
ARG IMG_REPO=pytorch

FROM pytorch/$IMG_REPO:$IMG_TAG

RUN apt-get -y update && apt-get -y install git gcc llvm-9-dev cmake libaio-dev vim wget

RUN git clone https://github.com/microsoft/DeepSpeed.git /tmp/DeepSpeed
RUN cd /tmp/DeepSpeed && DS_BUILD_OPS=1 ./install.sh -r
RUN pip install git+https://github.com/lucidrains/DALLE-pytorch.git

WORKDIR dalle
