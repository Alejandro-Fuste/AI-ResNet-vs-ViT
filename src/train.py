import argparse
import torch
from torchvision.models import resnet18, vit_b_16
from .datasets import build_loaders

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        m = resnet18(weights=None)      # you’ll load weights separately if fine-tuning
        m.fc.out_features = num_classes # or replace the head as needed
    elif model_name == "vit_b_16":
        m = vit_b_16(weights=None)
        m.heads.head.out_features = num_classes
    else:
        raise ValueError("model_name must be resnet18 or vit_b_16")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet18","vit_b_16"], required=True)
    ap.add_argument("--data",  default="data/Structured")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    loaders = build_loaders(args.data, model_name=args.model, batch_size=args.batch_size)
    num_classes = len(loaders["class_to_idx"])

    model = get_model(args.model, num_classes)
    # … optimizer, loss, training loop using loaders["train"]/["val"]

if __name__ == "__main__":
    main()
