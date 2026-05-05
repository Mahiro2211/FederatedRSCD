"""
Evaluate a change detection model on test data and produce full visualizations.

Usage:
    # Quick test with random model (no checkpoint needed):
    PYTHONPATH=. python tools/visualize_results.py

    # Evaluate a trained checkpoint:
    PYTHONPATH=. python tools/visualize_results.py --checkpoint saved_models/model_best.pth

    # Lite mode (fewer samples for faster iteration):
    PYTHONPATH=. python tools/visualize_results.py --max_test_samples 500 --n_samples 4
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from loguru import logger

from configs import create_model, get_dataset_configs
from data.collate_func import collate_func
from data.data_processing import CDDataset

from visualization import (
    plot_change_detection_comparison,
    plot_change_detection_overlay,
    plot_client_distribution,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_weight_similarity,
)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize model predictions on test set")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--model_name", type=str, default="BASE_Transformer")
    p.add_argument("--datasets", type=str, default="/home/dhm/dataset/")
    p.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--n_samples", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_test_samples", type=int, default=0, help="Max test samples (0=all)"
    )
    p.add_argument(
        "--eval_batches", type=int, default=0, help="Max eval batches (0=all)"
    )
    return p.parse_args()


def _load_model(args, checkpoint_path=None):
    model = create_model(args.model_name, args)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.model_name} | Params: {total_params:,}")

    ckpt_path = checkpoint_path or getattr(args, "checkpoint", None)
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        epoch = ckpt.get("epoch", "?")
        logger.info(f"Loaded checkpoint: {ckpt_path} (epoch={epoch})")
    else:
        logger.warning(
            "No checkpoint loaded - using RANDOM weights (for flow testing only)"
        )

    return model.to(args.device).eval()


def _build_test_data(ds_config, args):
    all_test_datasets = []
    ds_names_for_samples = []

    for name, info in ds_config.items():
        ds_path = info["path"]
        if not os.path.isdir(ds_path):
            logger.warning(f"Dataset path not found: {ds_path}, skipping")
            continue

        test_ds = CDDataset(
            root_dir=ds_path,
            split="test",
            img_size=args.img_size,
            is_train=False,
            label_transform="norm",
        )
        logger.info(f"  {name}: {len(test_ds)} test samples")
        all_test_datasets.append(test_ds)
        ds_names_for_samples.append((name, test_ds))

    if not all_test_datasets:
        raise RuntimeError("No test datasets found")

    full_test = ConcatDataset(all_test_datasets)

    max_test = getattr(args, "max_test_samples", 0)
    if max_test > 0 and len(full_test) > max_test:
        random.seed(args.seed)
        indices = random.sample(range(len(full_test)), max_test)
        full_test = Subset(full_test, indices)
        logger.info(f"Test set subsampled -> {max_test}")

    test_loader = DataLoader(
        full_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_func,
    )

    return test_loader, ds_names_for_samples


def _run_evaluation(model, test_loader, device, max_batches=0):
    all_labels = []
    all_probs = []
    confusion_matrix = torch.zeros(2, 2, dtype=torch.long, device=device)

    logger.info(f"Running evaluation (max_batches={max_batches or 'all'})...")
    with torch.no_grad():
        for batch_idx, (A, B, L, names) in enumerate(test_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                logger.info(f"  Reached {max_batches} batches, stopping")
                break

            A = A.to(device, non_blocking=True)
            B = B.to(device, non_blocking=True)
            L = L.to(device, non_blocking=True)

            dev_type = "cuda" if "cuda" in device else "cpu"
            with torch.autocast(device_type=dev_type, dtype=torch.float16):
                pred = model(A, B)

            last_pred = pred[-1] if isinstance(pred, (list, tuple)) else pred
            probs = F.softmax(last_pred, dim=1)[:, 1]
            pred_labels = last_pred.argmax(dim=1)

            L_sq = L.squeeze(1) if L.dim() == 4 else L
            mask = (L_sq >= 0) & (L_sq < 2)
            indices = 2 * L_sq[mask].long() + pred_labels[mask].long()
            confusion_matrix += torch.bincount(indices, minlength=4).reshape(2, 2)

            all_labels.append(L.cpu())
            all_probs.append(probs.cpu())

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {(batch_idx + 1) * A.size(0)} samples...")

    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    from utils.tools import cm2score

    metrics = cm2score(confusion_matrix.cpu().numpy())
    logger.info("=== Metrics ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics, confusion_matrix.cpu().numpy(), all_labels, all_probs


def _collect_samples(ds_names, n_samples, seed):
    sample_images = []

    random.seed(seed)
    for name, ds in ds_names:
        n = len(ds)
        pick = min(n_samples, n)
        for idx in random.sample(range(n), pick):
            sample_images.append(ds[idx])

    A = torch.stack([s["A"] for s in sample_images])
    B = torch.stack([s["B"] for s in sample_images])
    L = torch.stack([s["L"] for s in sample_images])
    names = [s["name"] for s in sample_images]
    return A, B, L, names


@torch.no_grad()
def _predict_samples(model, A, B, device):
    A = A.to(device)
    B = B.to(device)
    dev_type = "cuda" if "cuda" in device else "cpu"
    with torch.autocast(device_type=dev_type, dtype=torch.float16):
        pred = model(A, B)
    last_pred = pred[-1] if isinstance(pred, (list, tuple)) else pred
    pred_labels = last_pred.argmax(dim=1).cpu()
    probs = F.softmax(last_pred, dim=1)[:, 1].cpu()
    return pred_labels, probs


def run_visualization(args, model=None, ds_config=None, output_dir=None):
    """
    Core visualization pipeline. Can be called from main.py after training.

    Args:
        args: config namespace (needs .device, .datasets, .model_name, etc.)
        model: trained model (if None, loads from args.checkpoint or random init)
        ds_config: dataset config dict (if None, built from args.datasets)
        output_dir: where to save plots (if None, uses args.output_dir or auto)
    """
    if output_dir is None:
        output_dir = getattr(args, "output_dir", None)
        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./viz_results/viz_{ts}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Visualization output: {output_dir}")

    if ds_config is None:
        ds_config = get_dataset_configs(args.datasets)

    if model is None:
        model = _load_model(args)

    test_loader, ds_names = _build_test_data(ds_config, args)

    eval_batches = getattr(args, "eval_batches", 0)
    metrics, cm, all_labels, all_probs = _run_evaluation(
        model,
        test_loader,
        args.device,
        max_batches=eval_batches,
    )

    # 1. Confusion matrix
    plot_confusion_matrix(
        cm,
        class_names=["unchanged", "changed"],
        normalize=True,
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
    )
    logger.info("[1/7] Confusion matrix")

    # 2. ROC
    plot_roc_curve(
        all_labels, all_probs, save_path=os.path.join(output_dir, "roc_curve.png")
    )
    logger.info("[2/7] ROC curve")

    # 3. PR
    plot_pr_curve(
        all_labels, all_probs, save_path=os.path.join(output_dir, "pr_curve.png")
    )
    logger.info("[3/7] PR curve")

    # 4. Client distribution
    client_info = []
    dataset_sizes = []
    for name, info in ds_config.items():
        for i in range(info["n_clients"]):
            client_info.append(
                {
                    "client_id": len(client_info),
                    "dataset_name": name,
                    "sampler_config": info["sampler_configs"][i],
                }
            )
        ds_obj = CDDataset(
            root_dir=info["path"],
            split="train",
            img_size=args.img_size,
            is_train=False,
            label_transform="norm",
        )
        for ratio in info["data_ratios"]:
            dataset_sizes.append(int(len(ds_obj) * ratio))

    plot_client_distribution(
        client_info,
        dataset_sizes,
        save_path=os.path.join(output_dir, "client_distribution.png"),
    )
    logger.info("[4/7] Client distribution")

    # 5. Change detection comparison
    n_samp = getattr(args, "n_samples", 6)
    seed = getattr(args, "seed", 42)
    A, B, L, names = _collect_samples(ds_names, n_samp, seed)
    pred_labels, sample_probs = _predict_samples(model, A, B, args.device)

    plot_change_detection_comparison(
        img_A=A,
        img_B=B,
        label=L,
        pred=pred_labels,
        img_names=names,
        n_samples=len(names),
        save_path=os.path.join(output_dir, "change_detection_comparison.png"),
    )
    logger.info("[5/7] Change detection comparison")

    # 6. Overlay
    plot_change_detection_overlay(
        img_A=A,
        label=L,
        pred=pred_labels,
        n_samples=len(names),
        save_path=os.path.join(output_dir, "change_detection_overlay.png"),
    )
    logger.info("[6/7] Change detection overlay")

    # 7. Weight similarity
    state_dicts = [model.state_dict()] * len(ds_names)
    sd_labels = [f"{name}_model" for name, _ in ds_names]
    plot_weight_similarity(
        state_dicts,
        labels=sd_labels,
        global_model=model,
        save_path=os.path.join(output_dir, "weight_similarity.png"),
    )
    logger.info("[7/7] Weight similarity")

    logger.info(f"All visualizations saved to: {output_dir}")
    return metrics


def main():
    args = parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
