FROM ubuntu:16.04

MAINTAINER Vladimir Chernykh <vladimir.chernykh@phystech.edu>

## Base packages for ubuntu

RUN apt-get update --fix-missing && apt-get install -y \
    git \
    wget \
    bzip2 \
    vim \
    g++ \
    software-properties-common \
    apt-transport-https

## Download and install anaconda

RUN wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O /tmp/anaconda.sh

RUN /bin/bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh

## Add anaconda on path

ENV PATH /opt/conda/bin:$PATH

## Update conda version (otherwise following installations give strange error)

RUN conda update -n base conda

## Audio libs

RUN conda install -c conda-forge librosa
RUN pip install SpeechRecognition

## TF + Keras
RUN pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.7.0-cp36-cp36m-linux_x86_64.whl
RUN pip install Keras==2.1.5

## Visualization libs
RUN pip install progressbar2

## Mount point for processing folder

RUN mkdir /root/processing
WORKDIR /root/processing

## Copy neccessary files

COPY code code

## Default startup command

CMD bash -c "python code/run_server.py"
