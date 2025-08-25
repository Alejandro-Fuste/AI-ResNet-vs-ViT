#!/usr/bin/env bash
set -e
conda activate resnet-vs-vit

# Demo a few images (place 2–3 JPEGs in demo_inputs/)
python -m src.demo \
  --model resnet18 \
  --checkpoint ./outputs/resnet18/best.pt \
  --inputs ./demo_inputs \
  --img_size 224 \
  --save_dir ./outputs/demo_resnet

python -m src.demo \
  --model vit_b_16 \
  --checkpoint ./outputs/vit_b16/best.pt \
  --inputs ./demo_inputs \
  --img_size 224 \
  --save_dir ./outputs/demo_vit
