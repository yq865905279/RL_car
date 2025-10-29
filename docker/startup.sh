#!/bin/bash

export USER=root
# 启动 SSH 服务
/usr/sbin/sshd

# 启动 VNC 服务器
vncserver :1 -geometry 1280x800 -depth 24 -localhost no

# 防止容器退出
tail -f /dev/null

