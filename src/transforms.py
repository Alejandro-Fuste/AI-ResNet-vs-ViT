# src/transforms.py
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

def build_transforms(model_name: str):
    if model_name.lower() in {"resnet18", "resnet"}:
        weights = ResNet18_Weights.IMAGENET1K_V1
    elif model_name.lower() in {"vit_b_16", "vit", "vit-b-16"}:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
    else:
        raise ValueError("model_name must be 'resnet18' or 'vit_b_16'")
    return weights, weights.transforms()  # includes resize/center-crop to 224 + normalize
