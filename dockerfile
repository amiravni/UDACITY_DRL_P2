#FROM ubuntu:18.04
FROM nvidia/cudagl:10.1-devel-ubuntu18.04


RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get install -y software-properties-common 
RUN apt-get install -y git

RUN git clone https://github.com/udacity/deep-reinforcement-learning.git \
 && cd deep-reinforcement-learning/python \
 && pip install .

RUN git clone https://github.com/openai/gym.git \
 && cd gym \
 && pip install -e .

ENV TZ=Europe/Minsk
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y python3-tk
RUN pip install progressbar2

env QT_X11_NO_MITSHM 1


