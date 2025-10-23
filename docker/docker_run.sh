#!/bin/bash

# 容器名称变量
CONTAINER_NAME="rl_test"
image_name="rl_car:1.0"

# 检查容器是否已存在
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "容器已存在，正在启动..."
    docker start -i $CONTAINER_NAME
else
    echo "创建并启动新容器..."
    docker run --name $CONTAINER_NAME \
      --gpus all \
      -e DISPLAY=$DISPLAY \
      -e XAUTHORITY=$XAUTHORITY \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v $XAUTHORITY:$XAUTHORITY \
      --network host \
      --privileged \
      -v $HOME/Docker-Project/RL-car/$CONTAINER_NAME/workspace:/root/workspace \
      -v /dev:/dev \
      --ipc=host \
      -itd $image_name
fi