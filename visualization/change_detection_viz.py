import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_change_detection_comparison(
    img_A: np.ndarray | torch.Tensor,
    img_B: np.ndarray | torch.Tensor,
    label: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor | None = None,
    img_names: list[str] | None = None,
    save_path: str | None = None,
    n_samples: int = 4,
    figsize_per_sample: tuple[int, int] = (16, 4),
    dpi: int = 150,
):
    """
    Plot side-by-side comparison of pre-image, post-image, ground truth,
    and optionally prediction for change detection.

    Args:
        img_A: pre-event images, shape (N, 3, H, W) or (N, H, W, 3).
        img_B: post-event images, same shape convention.
        label: ground truth labels, shape (N, 1, H, W) or (N, H, W).
        pred: prediction labels, same shape as label. If None, shows 3 columns.
        img_names: optional image names for titles.
        save_path: if provided, save figure to this path.
        n_samples: number of samples to display.
        figsize_per_sample: figure width/height per sample row.
        dpi: figure dpi.
    """

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    img_A = to_numpy(img_A)
    img_B = to_numpy(img_B)
    label = to_numpy(label)
    if pred is not None:
        pred = to_numpy(pred)

    # normalize image tensors from [-1,1] to [0,1] if needed
    def normalize_img(img):
        if img.min() < 0:
            img = (img + 1) / 2.0
        return np.clip(img, 0, 1)

    # handle (N, 3, H, W) -> (N, H, W, 3)
    if img_A.ndim == 4 and img_A.shape[1] == 3:
        img_A = normalize_img(img_A.transpose(0, 2, 3, 1))
        img_B = normalize_img(img_B.transpose(0, 2, 3, 1))
    elif img_A.ndim == 4:
        img_A = normalize_img(img_A)
        img_B = normalize_img(img_B)

    # handle (N, 1, H, W) -> (N, H, W) for labels
    if label.ndim == 4:
        label = label.squeeze(1)
    if pred is not None and pred.ndim == 4:
        pred = pred.squeeze(1)

    # if pred has class logits (N, C, H, W), take argmax
    if pred is not None and pred.ndim == 4 and pred.shape[1] > 1:
        pred = pred.argmax(axis=1)

    n = min(n_samples, len(img_A))
    n_cols = 4 if pred is not None else 3
    col_titles = ["Image A (pre)", "Image B (post)", "Ground Truth"]
    if pred is not None:
        col_titles.append("Prediction")

    fig, axes = plt.subplots(
        n, n_cols, figsize=(figsize_per_sample[0], figsize_per_sample[1] * n), dpi=dpi
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(img_A[i])
        axes[i, 1].imshow(img_B[i])
        axes[i, 2].imshow(label[i], cmap="gray", vmin=0, vmax=1)
        if pred is not None:
            axes[i, 3].imshow(pred[i], cmap="gray", vmin=0, vmax=1)

        row_title = img_names[i] if img_names and i < len(img_names) else f"Sample {i}"
        axes[i, 0].set_ylabel(row_title, fontsize=9)

        for j in range(n_cols):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle("Change Detection Comparison", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_change_detection_overlay(
    img_A: np.ndarray | torch.Tensor,
    label: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor | None = None,
    save_path: str | None = None,
    n_samples: int = 4,
    dpi: int = 150,
):
    """
    Plot change detection overlay: ground truth (green) and prediction (red)
    overlaid on the pre-event image. Overlap appears yellow.

    Args:
        img_A: pre-event images, shape (N, 3, H, W) or (N, H, W, 3).
        label: ground truth, shape (N, 1, H, W) or (N, H, W).
        pred: prediction, same shape. If None, only GT overlay shown.
        save_path: if provided, save figure to this path.
        n_samples: number of samples.
        dpi: figure dpi.
    """

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    img_A = to_numpy(img_A)
    label = to_numpy(label)
    pred = to_numpy(pred) if pred is not None else None

    if img_A.ndim == 4 and img_A.shape[1] == 3:
        img_A = (np.clip((img_A + 1) / 2.0, 0, 1)).transpose(0, 2, 3, 1)
    if label.ndim == 4:
        label = label.squeeze(1)
    if pred is not None:
        if pred.ndim == 4 and pred.shape[1] > 1:
            pred = pred.argmax(axis=1)
        elif pred.ndim == 4:
            pred = pred.squeeze(1)

    n = min(n_samples, len(img_A))
    n_cols = 3 if pred is not None else 2
    fig, axes = plt.subplots(n, n_cols, figsize=(5 * n_cols, 5 * n), dpi=dpi)
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        base = img_A[i].copy()

        # GT overlay (green)
        gt_overlay = base.copy()
        mask_gt = label[i] > 0
        gt_overlay[mask_gt] = [0, 1, 0]
        blended_gt = 0.5 * base + 0.5 * gt_overlay

        axes[i, 0].imshow(base)
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(np.clip(blended_gt, 0, 1))
        axes[i, 1].set_title("GT Overlay (green)")

        if pred is not None:
            pred_overlay = base.copy()
            mask_pred = pred[i] > 0
            pred_overlay[mask_pred] = [1, 0, 0]

            combined = base.copy()
            combined[mask_gt] = [0, 1, 0]
            combined[mask_pred] = [1, 0, 0]
            both = mask_gt & mask_pred
            combined[both] = [1, 1, 0]

            blended_combined = 0.5 * base + 0.5 * combined
            axes[i, 2].imshow(np.clip(blended_combined, 0, 1))
            axes[i, 2].set_title("GT(green) + Pred(red)")

        for j in range(n_cols):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
