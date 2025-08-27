# src/demo.py
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, vit_b_16

from .utils.seeding import seed_all


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _infer_num_classes_from_state_dict(state_dict: dict, model_name: str) -> Optional[int]:
    """
    Try to infer num_classes from the checkpoint's classifier weight shape.
    Works for ResNet-18 (fc.weight) and ViT-B/16 (heads.head.weight).
    Returns None if it can't infer.
    """
    candidate_keys = []
    if model_name == "resnet18":
        candidate_keys = ["fc.weight", "module.fc.weight"]
    elif model_name == "vit_b_16":
        candidate_keys = ["heads.head.weight", "module.heads.head.weight"]

    for k in candidate_keys:
        w = state_dict.get(k, None)
        if w is not None:
            # torch Tensor or numpy array
            try:
                return int(w.shape[0])
            except Exception:
                pass
    return None


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
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


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """
    Loads a checkpoint. Accepts formats:
      - {'state_dict': ..., 'class_to_idx': ...}  (common)
      - {'model_state_dict': ...}
      - raw state_dict
    Returns (model, ckpt_dict).
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # Figure out the state_dict
    if isinstance(ckpt, dict) and any(k.endswith("weight") for k in ckpt.keys()):
        state_dict = ckpt  # raw state_dict
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}")

    # Load weights (strict=True is preferred; fall back to strict=False)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)

    return model, (ckpt if isinstance(ckpt, dict) else {})


def load_class_names(args, ckpt_meta: dict) -> List[str]:
    """
    Priority order:
      1) --class_names JSON/TXT file supplied by user
      2) ckpt['idx_to_class'] (list or dict) or inverse of ckpt['class_to_idx']
      3) fallback 'class_{i}'
    """
    # From --class_names
    if args.class_names:
        p = Path(args.class_names)
        if p.suffix.lower() in {".json"}:
            with open(p, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                # If dict of idx->name, convert to list ordered by idx
                names = [name for _, name in sorted(obj.items(), key=lambda kv: int(kv[0]))]
            else:
                names = list(obj)
            return names
        else:
            # TXT: one class per line
            with open(p, "r") as f:
                return [line.strip() for line in f if line.strip()]

    # From checkpoint metadata
    if isinstance(ckpt_meta, dict):
        if "idx_to_class" in ckpt_meta and ckpt_meta["idx_to_class"]:
            idx_to_class = ckpt_meta["idx_to_class"]
            if isinstance(idx_to_class, dict):
                return [name for _, name in sorted(idx_to_class.items(), key=lambda kv: int(kv[0]))]
            return list(idx_to_class)
        if "class_to_idx" in ckpt_meta and ckpt_meta["class_to_idx"]:
            c2i = ckpt_meta["class_to_idx"]
            inv = {i: c for c, i in c2i.items()}
            return [inv[i] for i in sorted(inv.keys())]

    # Fallback
    return [f"class_{i}" for i in range(args.num_classes)]


def collect_images(root: str) -> List[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in IMG_EXTS:
        return [root]
    imgs = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    imgs.sort()
    return imgs


def overlay_label(img_bgr: np.ndarray, text: str, prob: float) -> np.ndarray:
    """
    Draw a semi-transparent text box with 'text (p=0.97)' at top-left.
    """
    out = img_bgr.copy()
    label = f"{text} (p={prob:.2f})"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x, y = 10, 30
    # box behind text
    cv2.rectangle(out, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), (0, 0, 0), thickness=-1)
    cv2.addWeighted(out, 0.8, img_bgr, 0.2, 0, out)
    # text
    cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Overlay top-1 prediction on images.")
    ap.add_argument("--model", choices=["resnet18", "vit_b_16"], required=True)
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--inputs", required=True, help="Image file or folder containing images")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--class_names", default=None, help="Optional path to JSON/TXT with class names")
    ap.add_argument("--num_classes", type=int, default=15, help="Used if num_classes cannot be inferred")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (to infer num_classes first)
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(raw_ckpt, dict) and any(k.endswith("weight") for k in raw_ckpt.keys()):
        sd = raw_ckpt
        meta = {}
    elif isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        sd = raw_ckpt["state_dict"]
        meta = raw_ckpt
    elif isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        sd = raw_ckpt["model_state_dict"]
        meta = raw_ckpt
    else:
        raise ValueError("Unrecognized checkpoint format.")

    inferred_nc = _infer_num_classes_from_state_dict(sd, args.model)
    num_classes = inferred_nc if inferred_nc is not None else args.num_classes

    # Build model and load weights
    model = build_model(args.model, num_classes)
    model, _ = load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()

    class_names = load_class_names(args, meta)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    images = collect_images(args.inputs)
    assert images, f"No images found under: {args.inputs}"

    with torch.no_grad():
        for img_path in images:
            # For overlay we need the original image too (OpenCV BGR)
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[WARN] Could not read {img_path}")
                continue

            pil = Image.open(img_path).convert("RGB")
            x = tfm(pil).unsqueeze(0).to(device)  # [1,3,H,W]

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top1 = int(torch.argmax(probs).item())
            p = float(probs[top1].item())

            label = class_names[top1] if top1 < len(class_names) else f"class_{top1}"
            out = overlay_label(bgr, label, p)

            out_name = save_dir / f"{img_path.stem}_pred.jpg"
            cv2.imwrite(str(out_name), out)
            print(f"[OK] {img_path.name} → {label} (p={p:.3f}) → {out_name}")


if __name__ == "__main__":
    main()
