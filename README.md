# FederatedRSCD

Remote Sensing Change Detection with Federated Learning.

[English](readme_doc/readme_EN.md) | [简体中文](readme_doc/readme_CN.md)

## Introduction

A federated learning framework for remote sensing change detection, supporting multiple datasets, model architectures, and loss functions. The framework implements FedAvg for weight aggregation and provides a realistic Non-IID benchmark with data volume imbalance and sampler heterogeneity across clients.

![stars](https://img.shields.io/github/stars/Mahiro2211/FederatedRSCD)

## Requirements

- Python 3.10+
- PyTorch >= 2.0

```bash
pip install -r requirements.txt
```

## Supported Models

13 model architectures are available, selectable via `--model_name`:

| Model | Architecture | Parameters | Reference |
|---|---|---|---|
| `BASE_Transformer` | BIT (ResNet18 + Transformer) | 11.9M | Chen et al., "Remote Sensing Image Change Detection With Transformers" |
| `BASE_Transformer_s4_dd8` | BIT + decoder depth 8 | 12.4M | - |
| `BASE_Transformer_s4_dd8_dedim8` | BIT + decoder dim 8 | 11.9M | - |
| `SiamUnet_diff` | Siamese U-Net (difference) | 1.4M | Daudt et al., "Fully Convolutional Siamese Networks for Change Detection" |
| `SiamUnet_conc` | Siamese U-Net (concatenation) | 1.5M | Daudt et al. |
| `Unet` | FC-EF (early fusion) | 1.4M | Daudt et al. |
| `ChangeFormerV1` | Transformer encoder + conv decoder | 38.2M | Bandara et al., "ChangeFormer: A Transformer-Based Siamese Network" |
| `ChangeFormerV2` | + Transformer decoder | 32.7M | - |
| `ChangeFormerV3` | + Feature fuse | 32.7M | - |
| `ChangeFormerV4` | + Dual-branch encoder | 35.8M | - |
| `ChangeFormerV5` | + Deep encoder | 55.3M | - |
| `ChangeFormerV6` | + Lightweight encoder | 41.0M | - |
| `DTCDSCN` | Dual-task constrained SCN | 41.1M | "Building Change Detection Using a Dual Task Constrained Deep Siamese Convolutional Network" |

## Supported Loss Functions

4 loss functions are available, selectable via `--loss_type`:

| Loss | Description |
|---|---|
| `ce` | Cross Entropy (default) |
| `focal` | Focal Loss - addresses class imbalance |
| `dice` | Dice Loss - optimized for segmentation |
| `ce_dice` | CE + Dice combination (0.5 weight each) |

## Datasets

Pre-cropped datasets are organized as follows:

```
datasets/
├── LEVIR/          # LEVIR-CD
├── S2Looking/      # S2Looking
└── WHUCD/          # WHU-CD
```

### Client Partitioning

```
Summary
├── Total Datasets: 3
├── Total Clients: 8
└── Total Training Samples: 26,632

LEVIR (2 clients)
├── Client 1: 2,563 samples (60%), Random sampler
└── Client 2: 1,139 samples (40%), Weighted sampler

S2Looking (4 clients)
├── Client 3: 14,000 samples (50%), Random sampler
├── Client 4:  5,040 samples (30%), Sequential sampler
├── Client 5:  1,260 samples (15%), Random sampler
└── Client 6:    140 samples  (5%), Weighted sampler

WHUCD (2 clients)
├── Client 7: 1,245 samples (50%), Random sampler
└── Client 8: 1,245 samples (50%), Sequential sampler
```

This partitioning introduces both data volume imbalance and sampler heterogeneity, forming a realistic Non-IID federated learning benchmark.

### Data Preprocessing

Crop raw datasets to 256x256 patches:

```bash
# See data/crop_dataset.py for details
python -c "from data.crop_dataset import crop_levir, crop_s2looking, crop_whu; ..."
```

## Usage

### Quick Start

```bash
# Default: BASE_Transformer + CE loss
python main.py --datasets /path/to/datasets/
```

### Select Model

```bash
# Siamese U-Net (difference)
python main.py --model_name SiamUnet_diff

# ChangeFormer V6 with custom embed dimension
python main.py --model_name ChangeFormerV6 --embed_dim 256

# DTCDSCN
python main.py --model_name DTCDSCN
```

### Select Loss Function

```bash
# Focal loss (for imbalanced data)
python main.py --loss_type focal

# Dice loss (for segmentation)
python main.py --loss_type dice

# Combined CE + Dice
python main.py --loss_type ce_dice
```

### Full Arguments

```
Model Selection:
  --model_name {BASE_Transformer,BASE_Transformer_s4_dd8,BASE_Transformer_s4_dd8_dedim8,
                SiamUnet_diff,SiamUnet_conc,Unet,
                ChangeFormerV1,ChangeFormerV2,ChangeFormerV3,ChangeFormerV4,
                ChangeFormerV5,ChangeFormerV6,DTCDSCN}
                        Model architecture (default: BASE_Transformer)
  --embed_dim EMBED_DIM
                        Embedding dimension for ChangeFormerV5/V6 (default: 256)

Loss Function:
  --loss_type {ce,focal,dice,ce_dice}
                        Loss function type (default: ce)

Training:
  --batch_size BATCH_SIZE
                        Batch size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of federated rounds (default: 10)
  --num_client_epoch NUM_CLIENT_EPOCH
                        Local epochs per client per round (default: 2)
  --lr LR               Learning rate (default: 0.0001)
  --weight_decay WEIGHT_DECAY
                        Weight decay (default: 0.0001)

Federated Learning:
  --frac FRAC           Fraction of clients per round (default: 0.5)
  --niid / --no-niid    Non-IID data distribution (default: --niid)
  --n_shards N_SHARDS   Number of shards for partitioning (default: 20)

Data:
  --datasets DATASETS   Root directory of datasets
  --num_workers_dataloader NUM_WORKERS
                        Data loading workers per client (default: 4)

Other:
  --device DEVICE       Device (default: cuda:0)
  --eval_interval EVAL_INTERVAL
                        Evaluation interval in rounds (default: 1)
  --save_dir SAVE_DIR   Directory to save models (default: ./saved_models)
```

### Example Commands

```bash
# Lightweight model + focal loss for imbalanced data
python main.py --model_name SiamUnet_diff --loss_type focal --datasets /path/to/data

# ChangeFormerV5 with dice loss
python main.py --model_name ChangeFormerV5 --embed_dim 256 --loss_type dice

# DTCDSCN with combined loss, more rounds, higher participation
python main.py --model_name DTCDSCN --loss_type ce_dice --num_epochs 50 --frac 0.8

# BIT with deeper decoder
python main.py --model_name BASE_Transformer_s4_dd8 --loss_type ce --lr 0.0005
```

## Post-training Visualization

After training completes, use `visualize_all.sh` to evaluate the best model and generate all visualizations in one step:

```bash
# Auto-find latest checkpoint, generate all visualizations
./visualize_all.sh

# Specify a checkpoint and output directory
./visualize_all.sh -c saved_models/fed_train_20260505_231548/model_best.pth -o ./results

# Use a different model architecture
./visualize_all.sh -m ChangeFormerV6

# Quick sanity check with fewer samples
./visualize_all.sh --max-test-samples 200 --n-samples 3 --eval-batches 5
```

The script produces the following outputs in the output directory:

| File | Description |
|---|---|
| `confusion_matrix.png` | Normalized confusion matrix |
| `roc_curve.png` | ROC curve with AUC score |
| `pr_curve.png` | Precision-Recall curve |
| `client_distribution.png` | Federated client data distribution |
| `change_detection_comparison.png` | Side-by-side input/prediction/ground truth |
| `change_detection_overlay.png` | Prediction overlay on input images |
| `weight_similarity.png` | Model weight similarity heatmap |
| `training_curves.png` | Loss, IoU, F1, Precision/Recall over rounds (from log) |
| `metrics_history.json` | Raw per-round metrics parsed from training log |
| `summary.txt` | Text summary with configuration and final metrics |

### Full Script Options

```
  -c, --checkpoint PATH      Model checkpoint (auto-detect latest if omitted)
  -d, --datasets PATH        Dataset root directory
  -o, --output-dir PATH      Output directory for all visualizations
  -m, --model NAME           Model architecture (default: BASE_Transformer)
      --device DEVICE        Device, e.g. cuda:0 or cpu (auto-detect if omitted)
      --embed-dim N          Embedding dimension (default: 256)
      --img-size N           Input image size (default: 256)
      --batch-size N         Batch size for evaluation (default: 8)
      --n-samples N          Number of sample images (default: 6)
      --seed N               Random seed (default: 42)
      --max-test-samples N   Max test samples, 0=all (default: 0)
      --eval-batches N       Max eval batches, 0=all (default: 0)
```

## Project Structure

```
FederatedRSCD/
├── main.py                     # Entry point, FedTrain class
├── visualize_all.sh            # Post-training visualization pipeline script
├── loss.py                     # Loss functions (CE, Focal, Dice, CE+Dice)
├── assgin_ds.py                # Dataset loading entry
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── dataset_config.py       # Dataset & client partition configs
│   └── model_config.py         # Model configs + factory function
├── backbone/
│   ├── BaseTransformer.py      # BIT / BASE_Transformer + ResNet
│   ├── ChangeFormer.py         # ChangeFormer V1-V6
│   ├── ChangeFormerBaseNetworks.py
│   ├── DTCDSCN.py              # CDNet34
│   ├── SiamUnet_diff.py        # Siamese U-Net (difference)
│   ├── SiamUnet_conc.py        # Siamese U-Net (concatenation)
│   ├── Unet.py                 # FC-EF
│   ├── resnet.py               # Custom ResNet backbone
│   └── help_funcs.py           # Transformer, attention modules
├── data/
│   ├── crop_dataset.py         # Dataset preprocessing
│   ├── data_augment.py         # Data augmentation
│   ├── fed_allocator.py        # Federated data allocation
│   ├── fed_sampler.py          # Sampler factory
│   ├── collate_func.py         # Collate function
│   └── data_processing.py      # Dataset classes
├── utils/
│   ├── args.py                 # CLI argument parser
│   └── tools.py                # Metrics, display utilities
├── tools/
│   └── visualize_results.py    # Evaluation & visualization entry
├── visualization/
│   ├── training_curves.py      # Training metric curve plots
│   ├── confusion_matrix_viz.py # Confusion matrix plots
│   ├── roc_pr_curves.py        # ROC & PR curve plots
│   ├── client_distribution.py  # Client distribution plots
│   ├── change_detection_viz.py # Change detection comparison & overlay
│   └── weight_similarity.py    # Weight similarity heatmap
└── elements/
    └── image.png               # Architecture diagram
```

## Citation

```bibtex
@article{zhao2025fedrs,
  title={FedRS-Bench: Realistic Federated Learning Datasets and Benchmarks in Remote Sensing},
  author={Zhao, Haodong and Peng, Peng and Chen, Chiyu and Huang, Linqing and Liu, Gongshen},
  journal={arXiv preprint arXiv:2505.08325},
  year={2025}
}
```

## AI-assisted Development

This project was developed with the assistance of AI tools (GLM) for:
- Code structuring and refactoring
- Documentation drafting and polishing
- Debugging and design discussions

All model design, experiments, and final decisions were made by the author.
