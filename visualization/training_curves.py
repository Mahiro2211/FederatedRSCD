import os

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    metrics_history: dict[str, list[float]],
    save_path: str | None = None,
    title: str = "Training Metrics",
    figsize: tuple[int, int] = (16, 10),
    dpi: int = 150,
):
    """
    Plot training curves for loss, mIoU, mF1 and per-class metrics over rounds.

    Args:
        metrics_history: dict mapping metric names to per-round value lists.
            Supported keys: 'loss', 'miou', 'mf1', 'acc', 'iou_0', 'iou_1',
            'F1_0', 'F1_1', 'precision_0', 'precision_1', 'recall_0', 'recall_1'
        save_path: if provided, save figure to this path.
        title: figure title.
        figsize: figure size in inches.
        dpi: figure dpi.
    """
    if not metrics_history:
        raise ValueError("metrics_history is empty")

    n_rounds = len(next(iter(metrics_history.values())))
    rounds = np.arange(1, n_rounds + 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # loss curve
    ax = axes[0, 0]
    if "loss" in metrics_history:
        ax.plot(rounds, metrics_history["loss"], "b-o", markersize=3, label="Loss")
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # mIoU curve
    ax = axes[0, 1]
    iou_keys = [k for k in ["miou", "iou_0", "iou_1"] if k in metrics_history]
    if iou_keys:
        styles = {
            "miou": ("r-o", "mIoU"),
            "iou_0": ("g--s", "IoU (unchanged)"),
            "iou_1": ("m--^", "IoU (changed)"),
        }
        for key in iou_keys:
            ls, label = styles[key]
            ax.plot(rounds, metrics_history[key], ls, markersize=3, label=label)
        ax.set_xlabel("Round")
        ax.set_ylabel("IoU")
        ax.set_title("IoU Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # F1 curve
    ax = axes[1, 0]
    f1_keys = [k for k in ["mf1", "F1_0", "F1_1"] if k in metrics_history]
    if f1_keys:
        styles = {
            "mf1": ("r-o", "mF1"),
            "F1_0": ("g--s", "F1 (unchanged)"),
            "F1_1": ("m--^", "F1 (changed)"),
        }
        for key in f1_keys:
            ls, label = styles[key]
            ax.plot(rounds, metrics_history[key], ls, markersize=3, label=label)
        ax.set_xlabel("Round")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # precision / recall
    ax = axes[1, 1]
    pr_keys = [k for k in metrics_history if k.startswith(("precision_", "recall_"))]
    if pr_keys:
        markers = {"precision": "o", "recall": "s"}
        colors = {
            "precision_0": "tab:blue",
            "precision_1": "tab:cyan",
            "recall_0": "tab:orange",
            "recall_1": "tab:red",
        }
        for key in pr_keys:
            prefix = key.split("_")[0]
            ax.plot(
                rounds,
                metrics_history[key],
                marker=markers[prefix],
                markersize=3,
                label=key.replace("_", " "),
                color=colors.get(key, None),
                linestyle="--" if prefix == "recall" else "-",
            )
        ax.set_xlabel("Round")
        ax.set_ylabel("Score")
        ax.set_title("Precision & Recall")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
