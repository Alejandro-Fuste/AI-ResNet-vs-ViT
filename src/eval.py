# src/eval.py
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, vit_b_16

from .utils.seeding import seed_all
from .utils.metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_per_class_f1,
    compute_confusion,
    save_confusion_matrix_figure,
    topk_accuracies,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -----------------------------
# Helpers
# -----------------------------
def _infer_num_classes_from_state_dict(state_dict: dict, model_name: str) -> Optional[int]:
    """
    Try to infer num_classes from classifier weight shape in the checkpoint.
    Works for ResNet-18 (fc.weight) and ViT-B/16 (heads.head.weight).
    """
    keys = []
    if model_name == "resnet18":
        keys = ["fc.weight", "module.fc.weight"]
    elif model_name == "vit_b_16":
        keys = ["heads.head.weight", "module.heads.head.weight"]
    for k in keys:
        w = state_dict.get(k)
        if w is not None:
            try:
                return int(w.shape[0])
            except Exception:
                pass
    return None


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        m = resnet18(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    elif model_name == "vit_b_16":
        m = vit_b_16(weights=None)
        in_features = m.heads.head.in_features
        m.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("model_name must be 'resnet18' or 'vit_b_16'")
    return m


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict, dict]:
    """
    Load checkpoint flexibly:
      - raw state_dict
      - {'state_dict': ..., 'class_to_idx': ..., 'idx_to_class': ...}
      - {'model_state_dict': ...}
    Returns: (model, state_dict_used, checkpoint_meta_dict)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and any(k.endswith("weight") for k in ckpt.keys()):
        state_dict = ckpt
        meta = {}
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = ckpt
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}")

    # strict=True preferred; fall back to strict=False for minor key-name drift (e.g., DataParallel)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()
    return model, state_dict, meta


def load_class_names(meta: dict, dataset_classes: List[str], class_names_path: Optional[str]) -> List[str]:
    """
    Priority:
      1) --class_names (JSON list/dict or TXT one-per-line)
      2) meta['idx_to_class'] or inverse(meta['class_to_idx'])
      3) dataset-provided class names
    """
    # From file
    if class_names_path:
        p = Path(class_names_path)
        if p.suffix.lower() == ".json":
            with open(p, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                # idx->name dict
                return [name for _, name in sorted(obj.items(), key=lambda kv: int(kv[0]))]
            return list(obj)
        # TXT
        with open(p, "r") as f:
            return [line.strip() for line in f if line.strip()]

    # From checkpoint meta
    if isinstance(meta, dict):
        if "idx_to_class" in meta and meta["idx_to_class"]:
            if isinstance(meta["idx_to_class"], dict):
                return [name for _, name in sorted(meta["idx_to_class"].items(), key=lambda kv: int(kv[0]))]
            return list(meta["idx_to_class"])
        if "class_to_idx" in meta and meta["class_to_idx"]:
            c2i = meta["class_to_idx"]
            inv = {i: c for c, i in c2i.items()}
            return [inv[i] for i in sorted(inv.keys())]

    # Fallback to dataset names
    return list(dataset_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def forward_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_proba) where:
      - y_true: shape [N], int labels
      - y_proba: shape [N, C], softmax probabilities
    """
    all_targets: List[int] = []
    all_proba: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_proba.append(probs)
        all_targets.extend(y.numpy().tolist())

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_proba = np.concatenate(all_proba, axis=0)
    return y_true, y_proba


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict:
    y_true, y_proba = forward_pass(model, loader, device)
    y_pred = np.argmax(y_proba, axis=1)

    acc = compute_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1(y_true, y_pred)
    per_class = compute_per_class_f1(y_true, y_pred, class_names=class_names)

    cm = compute_confusion(y_true, y_pred, labels=list(range(len(class_names))), normalize=None)
    cm_norm = compute_confusion(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")

    topk = {}
    try:
        topk = topk_accuracies(y_true, y_proba, ks=(1, 5))
    except Exception:
        # Some environments may not have sklearn >= 1.0; ignore gracefully
        pass

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class,           # dict
        "topk": topk,                        # dict {1:..., 5:...} if available
        "confusion_matrix": cm.tolist(),     # raw counts
        "confusion_matrix_norm": cm_norm.tolist(),
        "class_names": class_names,
    }


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split and save metrics/figures.")
    ap.add_argument("--model", choices=["resnet18", "vit_b_16"], required=True)
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--data_root", required=True, help="Root containing train/ val/ test/")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)

    # outputs
    ap.add_argument("--metrics_json", required=True, help="Save overall metrics (JSON)")
    ap.add_argument("--metrics_csv", default=None, help="Optional: save per-class F1 to CSV")
    ap.add_argument("--confmat_png", required=True, help="Save confusion matrix (raw counts)")
    ap.add_argument("--confmat_norm_png", default=None, help="Optional: save normalized confusion matrix")
    ap.add_argument("--class_names", default=None, help="Optional JSON/TXT with class names")

    # optional: micro compute stats (params, throughput)
    ap.add_argument("--throughput_samples", type=int, default=0, help="If >0, measure throughput on this many images")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device)

    # Dataset / loader
    split_dir = Path(args.data_root) / args.split
    assert split_dir.exists(), f"Split directory not found: {split_dir}"

    tfm = build_transform(args.img_size)
    ds = datasets.ImageFolder(str(split_dir), transform=tfm)
    class_names_ds = list(ds.classes)  # ordered by folder-scan order
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model with a best guess for num_classes (dataset), then adjust if checkpoint says otherwise
    tmp_model = build_model(args.model, num_classes=len(class_names_ds))

    # Load checkpoint (may have different num_classes shape)
    tmp_model, state_dict, meta = load_checkpoint_into_model(tmp_model, args.checkpoint, device)

    # If the checkpoint's head shape implies a different num_classes, rebuild head accordingly (rare in eval if ds matches)
    inferred_nc = _infer_num_classes_from_state_dict(state_dict, args.model)
    if inferred_nc is not None and inferred_nc != len(class_names_ds):
        # Rebuild with inferred head size and reload
        tmp_model = build_model(args.model, inferred_nc)
        tmp_model, state_dict, meta = load_checkpoint_into_model(tmp_model, args.checkpoint, device)

    # Class names priority: user file > checkpoint meta > dataset folder names
    class_names = load_class_names(meta, class_names_ds, args.class_names)

    # Evaluate
    results = evaluate(tmp_model, loader, class_names, device)

    # Attach a couple of extras
    results["model"] = args.model
    results["checkpoint"] = str(Path(args.checkpoint).resolve())
    results["split"] = args.split
    results["params"] = int(count_params(tmp_model))

    # Optional: quick throughput test
    if args.throughput_samples and args.throughput_samples > 0:
        n = min(args.throughput_samples, len(ds))
        x = torch.stack([ds[i]()]()
