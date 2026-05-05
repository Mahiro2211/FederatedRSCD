import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def plot_weight_similarity(
    models: list[nn.Module] | list[dict[str, torch.Tensor]],
    labels: list[str] | None = None,
    global_model: nn.Module | dict[str, torch.Tensor] | None = None,
    save_path: str | None = None,
    title: str = "Model Weight Cosine Similarity",
    figsize: tuple[int, int] = (8, 7),
    dpi: int = 150,
    cmap: str = "RdYlGn",
):
    """
    Plot pairwise cosine similarity heatmap between client models
    (and optionally the global model).

    Args:
        models: list of client models (nn.Module) or state dicts.
        labels: display names for each model.
        global_model: optional global model to include in the comparison.
        save_path: if provided, save figure to this path.
        title: figure title.
        figsize: figure size in inches.
        dpi: figure dpi.
        cmap: matplotlib colormap.
    """
    state_dicts = []
    for m in models:
        if isinstance(m, nn.Module):
            state_dicts.append(m.state_dict())
        else:
            state_dicts.append(m)

    all_vecs = [_flatten_params(sd) for sd in state_dicts]

    if global_model is not None:
        if isinstance(global_model, nn.Module):
            global_vec = _flatten_params(global_model.state_dict())
        else:
            global_vec = _flatten_params(global_model)
        all_vecs.append(global_vec)

    n = len(all_vecs)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = _cosine_similarity(all_vecs[i], all_vecs[j])

    if labels is None:
        labels = [f"Client {i}" for i in range(len(state_dicts))]
    if global_model is not None:
        labels = labels + ["Global"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(sim_matrix, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{sim_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if abs(sim_matrix[i, j]) > 0.7 else "black",
                fontsize=9,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_weight_drift(
    client_models: list[nn.Module] | list[dict[str, torch.Tensor]],
    global_model: nn.Module | dict[str, torch.Tensor],
    labels: list[str] | None = None,
    save_path: str | None = None,
    title: str = "Client Weight Drift from Global",
    figsize: tuple[int, int] = (10, 5),
    dpi: int = 150,
):
    """
    Plot bar chart of each client model's distance/drift from the global model.

    Args:
        client_models: list of client models or state dicts.
        global_model: the global model or its state dict.
        labels: display names for each client.
        save_path: if provided, save figure to this path.
        title: figure title.
        figsize: figure size.
        dpi: figure dpi.
    """
    if isinstance(global_model, nn.Module):
        global_vec = _flatten_params(global_model.state_dict())
    else:
        global_vec = _flatten_params(global_model)

    drifts = []
    for m in client_models:
        sd = m.state_dict() if isinstance(m, nn.Module) else m
        vec = _flatten_params(sd)
        cos_sim = _cosine_similarity(vec, global_vec)
        drifts.append(1.0 - cos_sim)

    if labels is None:
        labels = [f"Client {i}" for i in range(len(drifts))]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(drifts)))
    bars = ax.bar(range(len(labels)), drifts, color=colors, edgecolor="white")

    for bar, drift in zip(bars, drifts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(drifts) * 0.02,
            f"{drift:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Cosine Distance (1 - similarity)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _flatten_params(state_dict: dict[str, torch.Tensor]) -> np.ndarray:
    return torch.cat([p.flatten().float() for p in state_dict.values()]).cpu().numpy()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
