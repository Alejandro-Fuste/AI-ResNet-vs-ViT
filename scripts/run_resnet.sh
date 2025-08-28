#!/usr/bin/env bash
set -euo pipefail

# --- Make 'conda activate' work inside non-interactive shells (Colab) ---
# Try common Conda locations/hooks in Colab
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

# --- Activate your env (same name you were using) ---
conda activate resnet-vs-vit

# --- Run from repo root and ensure folders exist ---
cd "$(dirname "$0")/.."
mkdir -p logs outputs

# --- Your original training command (unchanged flags) ---
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
