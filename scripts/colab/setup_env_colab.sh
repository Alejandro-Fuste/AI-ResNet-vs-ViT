#!/usr/bin/env bash
set -euo pipefail
# Colab-friendly environment setup (no conda)

# Run from repo root
cd "$(dirname "$0")/.."

# Make sure logs/outputs folders exist
mkdir -p logs outputs

# Upgrade pip toolchain and install project deps
python -m pip install --upgrade pip wheel setuptools
if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
else
  echo "[WARN] No requirements.txt found; skipping."
fi

# Quick sanity prints (optional)
python -c "import sys; print('Python:', sys.version)"
python -c "import torch; print('Torch:', getattr(__import__('torch'), '__version__', 'not installed'))" || true
