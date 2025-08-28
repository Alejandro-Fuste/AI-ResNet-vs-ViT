#!/usr/bin/env bash
set -euo pipefail

# --- Make 'conda activate' work inside non-interactive shells (Colab) ---
if [ -z "${CONDA_EXE:-}" ]; then
  if [ -f "/usr/local/etc/profile.d/conda.sh" ]; then
    . "/usr/local/etc/profile.d/conda.sh"
  elif [ -d "/usr/local/mambaforge" ] && [ -f "/usr/local/mambaforge/etc/profile.d/conda.sh" ]; then
    . "/usr/local/mambaforge/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "[ERR] Conda not found. Run the Colab setup cell to install Mambaforge." >&2
    exit 127
  fi
fi

# --- Activate your env ---
conda activate resnet-vs-vit

# --- Run from repo root and ensure folders exist ---
cd "$(dirname "$0")/.."
mkdir -p logs outputs

# --- Your original training command (unchanged flags) ---
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
