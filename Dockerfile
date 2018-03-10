FROM ubuntu:16.04

MAINTAINER Vladimir Chernykh <vladimir.chernykh@phystech.edu>

## Base packages for ubuntu

RUN apt-get update --fix-missing && apt-get install -y \
    git \
    wget \
    bzip2 \
    htop \
    vim \
    g++ \
    software-properties-common \
    apt-transport-https

## Download and install anaconda

RUN wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O /tmp/anaconda.sh

RUN /bin/bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh

## Add anaconda on path

ENV PATH /opt/conda/bin:$PATH

## Update conda version (otherwise following installations give strange error)

RUN conda update -n base conda

## Audio libs

RUN conda install -c conda-forge librosa

## Mount point for processing folder

RUN mkdir /root/processing
WORKDIR /root/processing

## Copy neccessary files

COPY code code

## Default startup command

CMD bash -c "python code/flask_server.py"
