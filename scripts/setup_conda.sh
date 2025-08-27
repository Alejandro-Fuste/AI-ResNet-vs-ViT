#!/usr/bin/env bash
set -e
conda env create -f environment.yml || conda create -y -n resnet-vs-vit python=3.10
conda activate cvsys
# If you created the env via create, install deps:
pip install -r requirements.txt
