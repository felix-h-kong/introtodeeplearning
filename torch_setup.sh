#!/bin/sh

sudo apt install python3.12-venv -y # assumes ubuntu 24.04

python3 -m venv env
source env/bin/activate
env/bin/pip install mitdeeplearning opencv-python