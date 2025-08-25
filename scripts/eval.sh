#!/usr/bin/env bash
set -e
conda activate resnet-vs-vit

# ResNet-18
python -m src.eval \
  --model resnet18 \
  --checkpoint ./outputs/resnet18/best.pt \
  --data_root ./data \
  --split test \
  --img_size 224 \
  --metrics_json ./outputs/resnet18/test_metrics.json \
  --confmat_png  ./outputs/resnet18/confusion_matrix.png

# ViT-B/16
python -m src.eval \
  --model vit_b_16 \
  --checkpoint ./outputs/vit_b16/best.pt \
  --data_root ./data \
  --split test \
  --img_size 224 \
  --metrics_json ./outputs/vit_b16/test_metrics.json \
  --confmat_png  ./outputs/vit_b16/confusion_matrix.png
