#!/bin/bash
set -e

# This script ensures that the workspace is populated with the application source
# on the first run if the mounted volume is empty.

APP_SOURCE_DIR="/app_source"
WORKSPACE_DIR="/root/workspace/RL_car"

# Check if the workspace directory is empty. `ls -A` lists all entries except for . and ..
# If the output is empty, the directory is considered empty.
if [ -z "$(ls -A $WORKSPACE_DIR)" ]; then
   echo "Workspace is empty. Initializing with application source..."
   # Copy the contents of the app source directory to the workspace.
   # The trailing dot on the source path is important to copy contents, not the directory itself.
   cp -a "$APP_SOURCE_DIR/." "$WORKSPACE_DIR/"
else
   echo "Workspace already contains files. Skipping initialization."
fi

# Initialize conda for the current shell
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Option to activate Qwen environment
# if [ "$ACTIVATE_QWEN" = "true" ]; then
#     echo "Activating Qwen conda environment..."
#     conda activate Qwen
# fi

# 创建X11配置
touch ~/.Xauthority

# 设置虚拟X服务器（如果没有X11连接）
if [ -z "$DISPLAY" ] || [ "$USE_XVFB" = "true" ]; then
    echo "Starting virtual X server with Xvfb..."
    Xvfb :99 -screen 0 1024x768x24 &
    export DISPLAY=:99
    sleep 2
fi

# 使用NVIDIA硬件加速
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450
# export GALLIUM_DRIVER=llvmpipe
# export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_DRI3_DISABLE=1
# 设置OGRE渲染模式
export OGRE_RTT_MODE=FBO

# 检查OpenGL支持
echo "Checking OpenGL support..."
glxinfo | grep -i "opengl" | head -3 || echo "OpenGL information not available"

# 配置音频
echo "Setting up audio..."

# 创建虚拟音频设备
if [ "$ENABLE_AUDIO" = "true" ]; then
    echo "Starting PulseAudio server..."
    mkdir -p /etc/alsa/conf.d
    echo "pcm.!default { type pulse }" > /etc/alsa/conf.d/99-pulse.conf
    echo "ctl.!default { type pulse }" >> /etc/alsa/conf.d/99-pulse.conf
    
    # 启动PulseAudio
    pulseaudio --start --log-target=syslog || echo "Failed to start PulseAudio"
    
    # 检查音频设备
    echo "Audio devices:"
    aplay -l || echo "No audio devices found"
fi

echo "Environment setup complete."

# Execute command passed to docker run
exec "$@"
