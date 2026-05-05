import os

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    normalize: bool = True,
    save_path: str | None = None,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6),
    dpi: int = 150,
    cmap: str = "Blues",
):
    """
    Plot a confusion matrix as a heatmap.

    Args:
        cm: confusion matrix of shape (n_classes, n_classes).
        class_names: display names for each class.
        normalize: if True, normalize rows to percentages.
        save_path: if provided, save figure to this path.
        title: figure title.
        figsize: figure size in inches.
        dpi: figure dpi.
        cmap: matplotlib colormap name.
    """
    cm = np.asarray(cm, dtype=np.float64)
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        display_cm = cm / row_sums
        fmt = ".2%"
        vmin, vmax = 0, 1
    else:
        display_cm = cm
        fmt = "d"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(display_cm, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    thresh = display_cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = display_cm[i, j]
            text = f"{val:{fmt}}" if normalize else f"{int(val)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=12,
            )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix_comparison(
    cms: dict[str, np.ndarray],
    class_names: list[str] | None = None,
    normalize: bool = True,
    save_path: str | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: int = 150,
):
    """
    Plot multiple confusion matrices side by side for comparison.

    Args:
        cms: dict mapping dataset/client names to confusion matrices.
        class_names: display names for each class.
        normalize: if True, normalize rows to percentages.
        save_path: if provided, save figure to this path.
        figsize: figure size. Auto-computed if None.
        dpi: figure dpi.
    """
    n = len(cms)
    if figsize is None:
        figsize = (6 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        axes = [axes]

    if class_names is None:
        n_classes = next(iter(cms.values())).shape[0]
        class_names = [str(i) for i in range(n_classes)]

    for ax, (name, cm) in zip(axes, cms.items()):
        cm = np.asarray(cm, dtype=np.float64)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            display_cm = cm / row_sums
            fmt = ".2%"
        else:
            display_cm = cm
            fmt = "d"

        im = ax.imshow(display_cm, interpolation="nearest", cmap="Blues")
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            title=name,
            ylabel="True",
            xlabel="Pred",
        )

        thresh = display_cm.max() / 2.0
        n_cls = display_cm.shape[0]
        for i in range(n_cls):
            for j in range(n_cls):
                val = display_cm[i, j]
                text = f"{val:{fmt}}" if normalize else f"{int(val)}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=10,
                )

    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
