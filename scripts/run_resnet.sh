#!/usr/bin/env bash
set -e
conda activate resnet-vs-vit
python -m src.train \
  --model resnet18 \
  --pretrained 1 \
  --data_root ./data \
  --img_size 224 \
  --batch_size 64 \
  --epochs 25 \
  --optimizer adamw \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --seed 42 \
  --outdir ./outputs/resnet18 \
  --log_csv ./logs/resnet18_train.csv
