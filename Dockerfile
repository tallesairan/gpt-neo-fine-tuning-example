FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}


##############################################################################
# Installation/Basic Utilities
##############################################################################

RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
        apt-get update && \
        apt-get install -y git && \
        git --version

##############################################################################
# Client Liveness & Uncomment Port 22 for SSH Daemon
##############################################################################
# Keep SSH client alive from server side
RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
RUN cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
        sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config

 

##############################################################################
# Python
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9
RUN apt-get install -y python3.9 python3.9-dev python3.9-venv python3-pip python3-wheel build-essential
#RUN rm -f /usr/bin/python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

RUN python -m pip install --upgrade pip
RUN python -V && pip -V
RUN python -m pip install pyyaml
RUN python -m pip install ipython

 
##############################################################################
# Some Packages
##############################################################################
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        libsndfile-dev \
        libcupti-dev \
        libjpeg-dev \
        libpng-dev \
        screen \
        libaio-dev \
        wget \
        curl \
        nano \
        htop  \
        atop  \
        bash  \
        zip  \
        unzip

##############################################################################
# PyYAML build issue
# https://stackoverflow.com/a/53926898
##############################################################################
RUN rm -rf /usr/lib/python3*/dist-packages/yaml && \
        rm -rf /usr/lib/python3*/dist-packages/PyYAML-* 
RUN python -m pip install psutil \
        yappi \
        cffi \
        ipdb \
        pandas \
        matplotlib \
        py3nvml \
        pyarrow \
        graphviz \
        astor \
        boto3 \
        tqdm \
        sentencepiece \
        msgpack \
        requests \
        pandas \
        sphinx \
        sphinx_rtd_theme \
        scipy \
        transformers \
        numpy \
        scikit-learn \
        nvidia-ml-py3 \
        cupy-cuda100

##############################################################################
# PyTorch
##############################################################################

RUN python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

#RUN ltt install torch torchvision
ENV TENSORBOARDX_VERSION=1.8
RUN python -m pip install tensorboardX==${TENSORBOARDX_VERSION}
RUN python -m pip install torchsummary
RUN python -m pip install accelerate
RUN python -m pip install protobuf==3.20.*
##############################################################################
## Add deepspeed user
###############################################################################
# Add a deepspeed user with user id 8877
#RUN useradd --create-home --uid 8877 deepspeed
RUN useradd -m -d /home/deepspeed --uid 1000 --shell /bin/bash deepspeed
RUN usermod -aG sudo deepspeed
RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # Change to non-root privilege
USER deepspeed
##############################################################################
# DeepSpeed
##############################################################################
RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
RUN cd ${STAGE_DIR}/DeepSpeed && \
        git checkout . && \
        git checkout master && \
        ./install.sh --pip_sudo
RUN rm -rf ${STAGE_DIR}/DeepSpeed
RUN python -c "import deepspeed; print(deepspeed.__version__)"
USER root
WORKDIR /
RUN mkdir app

RUN git clone https://github.com/tallesairan/gpt-neo-fine-tuning-example /app

RUN cd /app

COPY ./train /app



RUN cd /app && \
        wget "https://inference-datasets.s3.eu-central-1.amazonaws.com/nsfw-pt-br-dataset-test.csv.zip" && \
        unzip nsfw-pt-br-dataset-test.csv.zip && \
        cp nsfw-pt-br-dataset-test.csv dataset-filtred-10k.csv && \
        wget "https://inference-datasets.s3.eu-central-1.amazonaws.com/dataset-filtred.csv.zip" && \
        unzip dataset-filtred.csv.zip
