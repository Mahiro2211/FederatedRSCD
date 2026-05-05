import os

import matplotlib.pyplot as plt
import numpy as np


def plot_client_distribution(
    client_info: list[dict],
    dataset_sizes: list[int],
    save_path: str | None = None,
    title: str = "Client Data Distribution",
    figsize: tuple[int, int] = (12, 6),
    dpi: int = 150,
):
    """
    Plot bar chart showing sample counts per client, grouped by dataset.

    Args:
        client_info: list of dicts with keys 'client_id', 'dataset_name',
            'sampler_config'.
        dataset_sizes: actual sample counts per client (same length as
            client_info).
        save_path: if provided, save figure to this path.
        title: figure title.
        figsize: figure size in inches.
        dpi: figure dpi.
    """
    dataset_names = sorted(set(info["dataset_name"] for info in client_info))
    colors = plt.cm.Set2(np.linspace(0, 1, len(dataset_names)))
    color_map = {name: colors[i] for i, name in enumerate(dataset_names)}

    client_labels = [f"Client {info['client_id']}" for info in client_info]
    ds_names = [info["dataset_name"] for info in client_info]
    bar_colors = [color_map[name] for name in ds_names]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars = ax.bar(
        range(len(client_labels)),
        dataset_sizes,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, size in zip(bars, dataset_sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(dataset_sizes) * 0.01,
            str(size),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(client_labels)))
    ax.set_xticklabels(client_labels, rotation=45, ha="right")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title, fontsize=14, fontweight="bold")

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=color_map[name], label=name) for name in dataset_names
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_client_ratio_pie(
    client_info: list[dict],
    dataset_sizes: list[int],
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    dpi: int = 150,
):
    """
    Plot pie charts of data distribution, one per dataset.

    Args:
        client_info: list of dicts with 'client_id', 'dataset_name'.
        dataset_sizes: sample counts per client.
        save_path: if provided, save figure to this path.
        figsize: figure size in inches.
        dpi: figure dpi.
    """
    from collections import defaultdict

    dataset_clients: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for info, size in zip(client_info, dataset_sizes):
        dataset_clients[info["dataset_name"]].append((info["client_id"], size))

    n_datasets = len(dataset_clients)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, dpi=dpi)
    if n_datasets == 1:
        axes = [axes]

    for ax, (ds_name, clients) in zip(axes, dataset_clients.items()):
        labels = [f"Client {cid}" for cid, _ in clients]
        sizes = [s for _, s in clients]
        colors = plt.cm.Set3(np.linspace(0, 1, len(clients)))

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax.set_title(ds_name, fontsize=12, fontweight="bold")

    fig.suptitle("Data Distribution per Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
