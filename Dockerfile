FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y \
    python3.10 python3.11-distutils python3.10-dev libcupti-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py --force-reinstall && \
    rm -rf /var/lib/apt/lists/*
