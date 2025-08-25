#!/usr/bin/env bash
set -e
conda activate resnet-vs-vit
python -m src.train \
  --model vit_b_16 \
  --pretrained 1 \
  --data_root ./data \
  --img_size 224 \
  --batch_size 32 \
  --epochs 25 \
  --optimizer adamw \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --seed 42 \
  --outdir ./outputs/vit_b16 \
  --log_csv ./logs/vit_b16_train.csv
