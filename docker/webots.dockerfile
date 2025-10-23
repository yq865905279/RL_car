FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.0-runtime-ubuntu20.04
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# 预设keyboard-configuration配置以避免交互式安装
RUN echo 'keyboard-configuration keyboard-configuration/layout select US' | debconf-set-selections
RUN echo 'keyboard-configuration keyboard-configuration/layoutcode select us' | debconf-set-selections  
RUN echo 'keyboard-configuration keyboard-configuration/model select Generic 105-key (Intl) PC' | debconf-set-selections
RUN echo 'keyboard-configuration keyboard-configuration/modelcode select pc105' | debconf-set-selections
RUN echo 'keyboard-configuration keyboard-configuration/variant select English (US)' | debconf-set-selections
RUN echo 'keyboard-configuration keyboard-configuration/variantcode select ' | debconf-set-selections

# Change apt sources to USTC mirrors
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# Set timezone
RUN apt-get update && apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# Install basic packages
RUN apt-get install -y \
    curl \
    git \
    gnupg2 \
    lsb-release \
    sudo \
    vim \
    wget \
    software-properties-common \
    python3-pip \
    python3-dev \
    python3-distro \
    python3-yaml \
    keyboard-configuration \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 (only base package is reliably available)
# Note: python3.10-dev, python3.10-venv, python3.10-distutils may not be available in deadsnakes PPA
# Install Miniconda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm ~/miniconda3/miniconda.sh
RUN echo "source ~/miniconda3/bin/activate" >> ~/.bashrc

# Accept Conda Terms of Service
RUN ~/miniconda3/bin/conda config --set auto_update_conda false && \
    ~/miniconda3/bin/conda tos accept --override-channels --channel defaults

# Create conda environment
RUN ~/miniconda3/bin/conda create -y -n rl_car python=3.10

# Copy requirements.txt and install Python packages
COPY requirements.txt /root/workspace/requirements.txt
RUN ~/miniconda3/bin/conda run -n rl_car pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /root/workspace/requirements.txt

# Set the default shell to automatically activate the conda environment for interactive sessions
RUN echo "conda activate rl_car" >> ~/.bashrc

# Install X11 packages and audio support (runtime versions only)
RUN apt-get update && apt-get install -y \
    openssh-server \
    x11-apps \
    xauth \
    xterm \
    libgl1-mesa-glx \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libglu1-mesa \
    xvfb \
    libasound2 \
    libasound2-plugins \
    libasound2-data \
    alsa-base \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && cd /etc/apt/keyrings && wget -q https://cyberbotics.com/Cyberbotics.asc
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/Cyberbotics.asc] https://cyberbotics.com/debian binary-amd64/" | tee /etc/apt/sources.list.d/Cyberbotics.list
RUN apt update
RUN wget https://github.com/cyberbotics/webots/releases/download/R2023b/webots_2023b_amd64.deb && apt install ./webots_2023b_amd64.deb -y

RUN export WEBOTS_HOME=/usr/local/webots
RUN export PYTHONPATH=${WEBOTS_HOME}/lib/controller/python:${PYTHONPATH}
RUN export LD_LIBRARY_PATH=${WEBOTS_HOME}/lib/controller:${LD_LIBRARY_PATH}
RUN git clone https://github.com/yq865905279/RL_car.git /app_source

# Setup X11 forwarding
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# NVIDIA和OpenGL配置 - 启用NVIDIA驱动
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_REQUIRE_CUDA=cuda>=12.0

# RViz2和Gazebo渲染修复
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV MESA_GL_VERSION_OVERRIDE=4.5
ENV MESA_GLSL_VERSION_OVERRIDE=450
ENV OGRE_RTT_MODE=FBO
ENV LIBGL_DEBUG=verbose
ENV LIBGL_DRI3_DISABLE=1

# 音频配置
ENV ALSA_CONFIG_PATH=/etc/alsa/conf.d/99-pulse.conf
ENV PULSE_SERVER=unix:/run/user/1000/pulse/native
ENV SDL_AUDIODRIVER=pulseaudio

# Create a workspace directory
RUN mkdir -p /root/workspace
WORKDIR /root/workspace

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
