## docker容器
### 容器构建
``` bash
export CONTAINER_NAME="your-container-name"
export IMAGE_NAME="your-image-name"
docker run --name $CONTAINER_NAME \
      --gpus all \
      -e DISPLAY=$DISPLAY \
      -e XAUTHORITY=$XAUTHORITY \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v $XAUTHORITY:$XAUTHORITY \
      --network host \
      --privileged \
      -v $HOME/Docker-Project/RL-car/$CONTAINER_NAME/workspace:/root/workspace \ # 挂载工作目录
      -v /dev:/dev \
      --ipc=host \
      -itd $IMAGE_NAME
```