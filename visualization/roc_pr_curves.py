import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)


def plot_roc_curve(
    labels: np.ndarray | torch.Tensor,
    probs: np.ndarray | torch.Tensor,
    save_path: str | None = None,
    title: str = "ROC Curve",
    figsize: tuple[int, int] = (8, 6),
    dpi: int = 150,
):
    """
    Plot ROC curve for binary change detection.

    Args:
        labels: ground truth binary labels, shape (N,) or (N, H, W).
        probs: predicted probability for the change class, same shape.
        save_path: if provided, save figure to this path.
        title: figure title.
        figsize: figure size.
        dpi: figure dpi.
    """
    labels = _to_numpy(labels).flatten().astype(np.int32)
    probs = _to_numpy(probs).flatten().astype(np.float64)

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color="blue")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return auc


def plot_pr_curve(
    labels: np.ndarray | torch.Tensor,
    probs: np.ndarray | torch.Tensor,
    save_path: str | None = None,
    title: str = "Precision-Recall Curve",
    figsize: tuple[int, int] = (8, 6),
    dpi: int = 150,
):
    """
    Plot Precision-Recall curve for binary change detection.

    Args:
        labels: ground truth binary labels, shape (N,) or (N, H, W).
        probs: predicted probability for the change class, same shape.
        save_path: if provided, save figure to this path.
        title: figure title.
        figsize: figure size.
        dpi: figure dpi.
    """
    labels = _to_numpy(labels).flatten().astype(np.int32)
    probs = _to_numpy(probs).flatten().astype(np.float64)

    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(recall, precision, "r-", linewidth=2, label=f"PR (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.15, color="red")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return ap


def plot_roc_pr_combined(
    labels: np.ndarray | torch.Tensor,
    probs: np.ndarray | torch.Tensor,
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 6),
    dpi: int = 150,
):
    """
    Plot ROC and PR curves side by side.

    Args:
        labels: ground truth binary labels.
        probs: predicted probability for the change class.
        save_path: if provided, save figure to this path.
        figsize: figure size.
        dpi: figure dpi.
    """
    labels = _to_numpy(labels).flatten().astype(np.int32)
    probs = _to_numpy(probs).flatten().astype(np.float64)

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    ax1.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.fill_between(fpr, tpr, alpha=0.15, color="blue")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(recall, precision, "r-", linewidth=2, label=f"AP = {ap:.4f}")
    ax2.fill_between(recall, precision, alpha=0.15, color="red")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return auc, ap


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)
