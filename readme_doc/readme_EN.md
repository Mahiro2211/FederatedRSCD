# Federated Learning for Remote Sensing Change Detection

A **Transformer-based federated learning system** for remote sensing image change detection.
The system supports **multi-client parallel training**, **automatic model aggregation**, and **comprehensive performance evaluation**.

---

## Training

```bash
# Activate environment
conda activate wslpython310

# Single-process training (recommended for production)
python main.py \
  --batch_size 16 \
  --num_epochs 50 \
  --num_client_epoch 5 \
  --lr 0.0001 \
  --frac 0.5 \
  --save_dir ./my_experiments
```

---

## Command-Line Arguments

| Argument                   | Default          | Description                                                |
| -------------------------- | ---------------- | ---------------------------------------------------------- |
| `--batch_size`             | 8                | Batch size                                                 |
| `--num_epochs`             | 20               | Number of federated training rounds (communication rounds) |
| `--num_client_epoch`       | 5                | Number of local training epochs per client                 |
| `--lr`                     | 0.0001           | Learning rate                                              |
| `--frac`                   | 0.5              | Fraction of clients participating in each round            |
| `--use_parallel`           | False            | Enable multi-process parallel training                     |
| `--n_workers`              | 4                | Number of parallel worker processes                        |
| `--num_workers_dataloader` | 4                | Number of DataLoader worker processes                      |
| `--save_dir`               | `./saved_models` | Directory to save trained models                           |
| `--eval_interval`          | 1                | Evaluation interval (number of rounds between evaluations) |

---


## Frequently Asked Questions

### How to adjust the number of parallel training processes?

```bash
# Use 4 processes (recommended)
python main.py --n_workers 4

# Use 8 processes (if sufficient CPU cores are available)
python main.py --n_workers 8

# Disable parallel training
python main.py  # without --use_parallel
```

---

### How to speed up data loading?

```bash
# Increase DataLoader workers
python main.py --num_workers_dataloader 8
```

**Recommendations:**

* Training: `4–8`
* Testing: `2–4`

---

### How to run testing only?

Modify `main.py` to skip training and directly perform evaluation:

```python
# In the main function
Trainer.load_model("path/to/model.pth")
test_metrics = Trainer.test()
```

---

### Issues with multi-process training

When using CUDA under WSL/Linux, multi-process training may encounter issues.

**Recommended solutions:**

* Use single-process training (default)
* If using multi-process training, set `--num_workers_dataloader 0`
* Or use single-process training with multi-process DataLoader
  (`--num_workers_dataloader 4`)

---

## Technical Details

### 1. Federated Learning Workflow

```
Start Training
    ↓
Initialize Global Model
    ↓
Repeat for num_epochs rounds
    ↓
Randomly select clients (frac × n_clients)
    ↓
Train clients (parallel or sequential)
    ↓
Aggregate client models (FedAvg)
    ↓
Update global model
    ↓
Periodic evaluation (every eval_interval rounds)
    ↓
Save best model
    ↓
End
```

---

### 2. Automatic Mixed Precision (AMP)

This project uses **PyTorch Automatic Mixed Precision** to accelerate training:

```python
with torch.autocast(device_type=self.args.device, dtype=torch.float16):
    pred = client_model(A, B)
    loss = nllloss(pred[0].contiguous(), Label)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**

* ~50% GPU memory reduction
* ~2× training speedup
* Minimal impact on model accuracy

---

## Dataset Configuration

Configure datasets inside the `main()` function in `main.py`:

```python
ds_name = {
    "dataset_name": {
        "path": "/path/to/dataset",   # Dataset path
        "n_clients": 2,               # Number of clients
        "data_ratios": [0.6, 0.4],    # Data split ratios
        "sampler_configs": [          # Sampler configuration
            {"type": "random", "shuffle": True},
            {"type": "weighted", "shuffle": True, "weights": None},
        ],
    },
}
```

### Supported Sampler Types

* `random`: Random sampling (**recommended**)
* `sequential`: Sequential sampling
* `weighted`: Weighted sampling

---

## References

* PyTorch Automatic Mixed Precision
  [https://pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
* Federated Learning (FedAvg)
  [https://arxiv.org/abs/1602.05629](https://arxiv.org/abs/1602.05629)
* Weights & Biases Documentation
  [https://docs.wandb.ai/](https://docs.wandb.ai/)

---

## Changelog

### 2026-01-25

* ✅ Dataset preprocessing completed

### 2026-01-26

* ✅ Codebase merged into a single unified file
* ✅ Added Rich progress bar with detailed training visualization
* ✅ Client-level epoch and batch progress tracking
* ✅ Fixed multi-process training issues
* ✅ Improved testing pipeline
* ✅ Updated documentation
