FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    build-essential \
    ffmpeg \
    openssh-server \
    tmux \
    g++ \
    htop \
    curl \
    git \
    tar \
    python3-pip \
    python3-numpy \
    python3-scipy \
    net-tools \
    nano \
    unzip \
    vim \
    wget \
    xpra \
    xvfb \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
            python3.9-venv \
            python3.9-dev 

ENV VIRTUAL_ENV=venv
RUN python3.9 -m venv /opt/$VIRTUAL_ENV
ENV PATH /opt/$VIRTUAL_ENV/bin:$PATH

WORKDIR /home/workdir
COPY . .
RUN pip install wheel jupyter Cython torch torchvision  \
    && pip install -r ./requirements/requirements_dev.txt 
#    && pip install -e .

# for jupyter ssh tensorboard
# https://dongkwan-kim.github.io/blogs/tensorboard-in-a-docker-container/
EXPOSE 6969 22 9696
RUN python -m ipykernel install --name=$VIRTUAL_ENV

