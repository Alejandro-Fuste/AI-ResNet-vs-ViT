# AI-ResNet-vs-ViT

## Dataset
Place the Kaggle Human Action Recognition dataset in ./data with subfolders `train/` and `test/`. (15 classes, ~12.6k images, train/test split provided.)

## Quick start
bash scripts/setup_conda.sh

## Train
bash scripts/run_resnet.sh
bash scripts/run_vit.sh

## Evaluate on test (saves JSON + confusion matrices)
bash scripts/eval.sh

## Live demo (saves overlaid predictions)
bash scripts/demo.sh

## Repro notes
- environment.yml captures exact conda env.
- logs/*.csv contain epoch-by-epoch training traces.
- outputs/* contain best checkpoints and figures.
