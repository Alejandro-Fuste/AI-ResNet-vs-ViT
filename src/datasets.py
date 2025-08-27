from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from .transforms import build_transforms

def build_loaders(data_root: str,
                  model_name: str,
                  batch_size: int = 64,
                  val_split: float = 0.10,
                  num_workers: int = 4,
                  pin_memory: bool = True,
                  seed: int = 42):
    data_root = Path(data_root)
    weights, common_tf = build_transforms(model_name)

    train_dir = data_root / "train"
    test_dir  = data_root / "test"

    train_ds_full = datasets.ImageFolder(str(train_dir), transform=common_tf)
    test_ds       = datasets.ImageFolder(str(test_dir),  transform=common_tf)

    g = torch.Generator().manual_seed(seed)
    n_total = len(train_ds_full)
    n_val   = int(val_split * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(train_ds_full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    class_to_idx = train_ds_full.dataset.class_to_idx if hasattr(train_ds_full, "dataset") else train_ds_full.class_to_idx

    return {
        "weights": weights,
        "transforms": common_tf,
        "class_to_idx": class_to_idx,
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
        "sizes": {"train": n_train, "val": n_val, "test": len(test_ds)}
    }
