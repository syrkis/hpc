FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y python3.11 python3.11-distutils libglfw3-dev git

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py --force-reinstall && \ 
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3.11 -m pip install -r requirements.txt

RUN python3.11 -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN python3.11 -m pip install tensorflow_datasets opencv-python pycocotools

RUN git clone https://github.com/syrkis/syrkis.git

RUN python3.11 -m pip install -e syrkis

ENV PYGLFW_PREVIEW=1

RUN curl https://ollama.ai/install.sh | sh

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
