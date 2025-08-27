# src/utils/metrics.py
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Overall accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Macro-averaged F1 across classes."""
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_per_class_f1(y_true: Sequence[int], y_pred: Sequence[int], class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Returns a dict of per-class F1: {class_name: f1}.
    If class_names is None, uses stringified class indices.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # report contains per-class entries keyed by label; filter numeric keys
    per_class = {}
    # Gather class labels in the order present in report (skipping non-class keys)
    labels = [k for k in report.keys() if k.isdigit()]
    for k in labels:
        idx = int(k)
        name = class_names[idx] if (class_names and idx < len(class_names)) else k
        per_class[str(name)] = float(report[k]["f1-score"])
    return per_class


def compute_confusion(y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[Iterable[int]] = None, normalize: Optional[str] = None) -> np.ndarray:
    """
    Confusion matrix (optionally normalized: 'true', 'pred', or 'all').
    """
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


def save_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """
    Plots and optionally saves the confusion matrix image.
    """
    if normalize and cm.max() > 1.0:
        # convert to row-normalized if a raw count matrix was provided
        with np.errstate(all="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # annotate cells
    thresh = (cm.max() + cm.min()) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def topk_accuracies(y_true: np.ndarray, y_proba: np.ndarray, ks: Tuple[int, ...] = (1, 5)) -> Dict[int, float]:
    """
    Compute top-k accuracies from probability or logit array of shape [N, C].
    """
    out = {}
    for k in ks:
        out[k] = float(top_k_accuracy_score(y_true, y_proba, k=k, labels=list(range(y_proba.shape[1]))))
    return out
