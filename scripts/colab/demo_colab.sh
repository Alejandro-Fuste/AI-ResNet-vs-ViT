#!/usr/bin/env bash
set -euo pipefail
# Simple demo runner in Colab without conda

cd "$(dirname "$0")/.."
mkdir -p outputs/demo_resnet outputs/demo_vit

# Expect a few JPEGs under ./demo_inputs
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
