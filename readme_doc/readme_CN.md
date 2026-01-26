# 联邦学习遥感图像变化检测

基于Transformer的联邦学习遥感图像变化检测系统，支持多客户端并行训练、自动模型聚合和详细性能评估。

### 训练

```bash
# 激活环境
conda activate wslpython310

# 单进程训练（推荐用于生产环境）
python main.py \
  --batch_size 16 \
  --num_epochs 50 \
  --num_client_epoch 5 \
  --lr 0.0001 \
  --frac 0.5 \
  --save_dir ./my_experiments
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 8 | 批大小 |
| `--num_epochs` | 20 | 联邦学习训练轮数（通信轮数） |
| `--num_client_epoch` | 5 | 每个客户端本地训练的轮数 |
| `--lr` | 0.0001 | 学习率 |
| `--frac` | 0.5 | 每轮参与训练的客户端比例 |
| `--use_parallel` | False | 是否使用多进程并行训练 |
| `--n_workers` | 4 | 并行训练的进程数 |
| `--num_workers_dataloader` | 4 | 数据加载器的工作进程数 |
| `--save_dir` | ./saved_models | 模型保存目录 |
| `--eval_interval` | 1 | 评估间隔（每几轮评估一次） |

## 代码结构

```
fed_rscd/
├── main.py      # 完整的联邦学习训练系统（推荐使用）
├── run_test.sh          # 测试运行脚本
├── README.md            # 本文件
├── loss.py              # 损失函数
├── assgin_ds.py         # 数据集分配
├── utils/               # 工具函数
│   └── args.py          # 训练参数
├── backbone/            # 模型架构
│   └── ...
└── data/                # 数据处理
    └── ...
```

## 常见问题

### 如何调整并行训练的进程数？

```bash
# 使用4个进程（推荐）
python main.py --n_workers 4

# 使用8个进程（如果CPU核心数充足）
python main.py --n_workers 8

# 禁用并行训练
python main.py （不使用--use_parallel）
```

### 如何调整数据加载速度？

```bash
# 增加数据加载器工作进程数
python main.py --num_workers_dataloader 8
```

**建议**：
- 训练时：设置为4-8
- 测试时：设置为2-4


### 如何只进行测试？

修改`main.py`，跳过训练直接测试：

```python
# 在main函数中
Trainer.load_model("path/to/model.pth")
test_metrics = Trainer.test()
```

### 6. 多进程训练遇到问题怎么办？

在WSL/Linux环境下使用CUDA时，多进程训练可能遇到问题。建议：

- 使用单进程训练（默认）
- 如果使用多进程，设置`--num_workers_dataloader 0`
- 或者使用单进程 + DataLoader多进程（`--num_workers_dataloader 4`）

## 技术细节

### 1. 联邦学习流程

```
开始训练
    ↓
加载全局模型
    ↓
循环 num_epochs 轮
    ↓
随机选择客户端 (frac * n_clients)
    ↓
并行/顺序训练客户端
    ↓
聚合客户端模型 (FedAvg)
    ↓
更新全局模型
    ↓
定期评估 (每 eval_interval 轮)
    ↓
保存最佳模型
    ↓
结束
```

### 2. 自动混合精度训练（AMP）

使用PyTorch的自动混合精度训练：

```python
with torch.autocast(device_type=self.args.device, dtype=torch.float16):
    pred = client_model(A, B)
    loss = nllloss(pred[0].contiguous(), Label)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**优点**：
- 减少显存占用（约50%）
- 加速计算（约2x）
- 保持训练精度


## 数据集配置

在`main.py`的`main()`函数中配置数据集：

```python
ds_name = {
    "数据集名称": {
        "path": "/path/to/dataset",  # 数据集路径
        "n_clients": 2,  # 分配到2个客户端
        "data_ratios": [0.6, 0.4],  # 数据分配比例
        "sampler_configs": [  # 采样器配置
            {"type": "random", "shuffle": True},
            {"type": "weighted", "shuffle": True, "weights": None},
        ],
    },
}
```

### 采样器类型
- `random`: 随机采样（推荐）
- `sequential`: 顺序采样
- `weighted`: 加权采样

## 参考资源

- PyTorch混合精度训练：https://pytorch.org/docs/stable/amp.html
- 联邦学习论文(FedAvg)：https://arxiv.org/abs/1602.05629
- WandB文档(日志系统)：https://docs.wandb.ai/

## 更新日志

### 2026-01-25
- ✅ 处理数据集

### 2026-01-26
- ✅ 完成代码合并，创建单文件完整版
- ✅ 添加Rich进度条，显示详细的训练进度
- ✅ 支持显示每个客户端的epoch和batch级别进度
- ✅ 修复多进程训练问题
- ✅ 完善测试功能
- ✅ 更新文档

