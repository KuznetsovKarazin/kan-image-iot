#!/bin/sh

#
# Environment prepare for TFLite Runtime
# Raspian OS Bookworm
#
sudo apt update
sudo apt install -y python3-numpy python3-opencv python3-pil
sudo apt install -y libopenblas-dev libatlas-base-dev
python3 -m venv --system-site-packages tflite-env
source tflite-env/bin/activate
pip3 install tflite-runtime
