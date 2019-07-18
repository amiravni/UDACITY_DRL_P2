#!/bin/bash
xhost +
nvidia-docker run --name=udrl2 -it --ipc=host -v ~/UDACITY_DRL_P2:/UDACITY_DRL_P2 -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority --privileged --net=host udrl_docker
