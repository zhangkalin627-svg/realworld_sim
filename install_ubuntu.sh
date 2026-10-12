#!/bin/bash
# 一键安装（Ubuntu 22.04 / 24.04 桌面版，x86_64 或 ARM64 都行）
set -e
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip fonts-noto-cjk libxcb-cursor0 libxkbcommon-x11-0 libegl1
# v0.6 电路仿真需要 ngspice（BSD-3 许可证，作为独立外部程序调用）
sudo apt-get install -y ngspice
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "安装完成。运行： ./run.sh"
