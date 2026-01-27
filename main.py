import copy
import os
import time
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from loss import cross_entropy
from train import train_client_worker
from utils.tools import display_client_info, get_all_metrics


class FedTrain:
    """
    联邦学习训练类

    实现联邦学习的完整训练流程，包括：
    - 客户端本地训练（支持多进程并行）
    - 模型权重聚合（FedAvg）
    - 全局模型评估（包含详细指标和可视化）
    - 模型保存和加载
    """

    def __init__(
        self, args, model, train_loader: list, test_loader: dict, n_clients: int
    ):
        """
        初始化联邦学习训练器

        Args:
            args: 训练配置参数
            model: 全局模型
            train_loader: 各客户端训练数据加载器列表 [dataloader0, dataloader1, ...]
            test_loader: 测试数据加载器字典 {dataset_name: dataloader}
            n_clients: 客户端总数
        """
        self.args = args
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args.n_clients = n_clients
        self.max_result = {
            "acc": 0.0,
            "loss": 0.0,
            "miou": 0.0,
            "iou_0": 0.0,
            "mf1": 0.0,
            "F1_0": 0.0,
            "F1_1": 0.0,
            "recall_0": 0.0,
            "recall_1": 0.0,
            "iou_1": 0.0,
            "precision_0": 0.0,
            "precision_1": 0.0,
        }

        # 使用DataParallel进行多GPU并行加速（如果可用）
        if torch.cuda.device_count() > 1 and not args.device.startswith("cpu"):
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)

        self.scaler = GradScaler()

        self.save_dir = os.path.join(
            args.save_dir, f"fed_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"模型和结果将保存到: {self.save_dir}")

        try:
            import wandb

            self.wandb = wandb
            logger.info("WandB已初始化")
        except ImportError:
            self.wandb = None
            logger.warning("WandB未安装，将跳过日志记录")

        if self.wandb is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            self.wandb.config.update(
                {
                    "model_total_params": total_params,
                    "model_trainable_params": trainable_params,
                    "num_gpus": torch.cuda.device_count()
                    if torch.cuda.is_available()
                    else 0,
                    "device": args.device,
                }
            )

    def train_client(self, model, dataloader, client_idx, progress=None):
        """
        在单个客户端上进行本地训练（单进程版本）

        Args:
            model: 客户端初始模型（全局模型的副本）
            dataloader: 客户端训练数据加载器
            client_idx: 客户端索引

        Returns:
            tuple: (训练后的模型, 平均损失)
        """
        client_model = copy.deepcopy(model)
        client_model.train()

        optimizer = torch.optim.Adam(
            client_model.parameters(),
            lr=self.args.lr,
            betas=self.args.betas,
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
        )

        client_scaler = GradScaler()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.args.num_client_epoch):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_batches = 0

            for A, B, Label, _ in dataloader:
                A = A.contiguous().to(self.args.device, non_blocking=True)
                B = B.contiguous().to(self.args.device, non_blocking=True)
                Label = Label.contiguous().to(self.args.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self.args.device, dtype=torch.float16):
                    pred = client_model(A, B)
                    loss = cross_entropy(pred[0].contiguous(), Label)

                client_scaler.scale(loss).backward()
                client_scaler.step(optimizer)
                client_scaler.update()

                total_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                epoch_batches += 1

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0

            logger.info(
                f"客户端 {client_idx} - Epoch {epoch + 1}/{self.args.num_client_epoch} 完成，损失: {avg_epoch_loss:.4f}, 耗时: {epoch_time:.2f}秒"
            )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return client_model, avg_loss

    def train_clients_parallel(self, selected_client_indices):
        """
        使用多进程并行训练多个客户端

        Args:
            selected_client_indices: 选中的客户端索引列表

        Returns:
            tuple: (客户端模型状态字典列表, 客户端损失列表)
        """
        import multiprocessing as mp

        global_state_dict = self.model.state_dict()

        client_args = []
        for idx in selected_client_indices:
            client_args.append(
                (
                    copy.deepcopy(global_state_dict),
                    idx,
                    self.args,
                    idx,
                    self.train_loader,
                )
            )

        n_workers = min(
            self.args.n_workers if hasattr(self.args, "n_workers") else cpu_count(),
            len(selected_client_indices),
        )
        logger.info(
            f"使用 {n_workers} 个进程并行训练 {len(selected_client_indices)} 个客户端"
        )

        ctx = mp.get_context("spawn")

        with ctx.Pool(processes=n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(train_client_worker, client_args),
                    total=len(client_args),
                    desc="并行训练客户端",
                )
            )

        client_models = []
        client_losses = []

        for i, (state_dict, loss) in enumerate(results):
            client_models.append(state_dict)
            client_losses.append(loss)
            logger.info(f"  客户端 {selected_client_indices[i]} 训练损失: {loss:.4f}")

        return client_models, client_losses

    def average_weights(self, clients_model: list, client_weights=None):
        """
        使用FedAvg算法聚合客户端模型权重

        Args:
            clients_model: 客户端模型状态字典列表 [state_dict1, state_dict2, ...]
            client_weights: 客户端权重列表（可选），如果不提供则使用平均权重

        Returns:
            dict: 聚合后的全局模型状态字典
        """
        if not clients_model:
            logger.warning("没有客户端模型需要聚合")
            return self.model.state_dict()

        # 计算每个客户端的权重
        if client_weights is None:
            client_weights = [1.0 / len(clients_model)] * len(clients_model)
        else:
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # 初始化聚合后的权重字典
        avg_weights = clients_model[0].copy()

        # 对每个参数进行加权平均
        for key in avg_weights.keys():
            avg_weights[key] = avg_weights[key] * client_weights[0]

            for i in range(1, len(clients_model)):
                avg_weights[key] += clients_model[i][key] * client_weights[i]

        return avg_weights

    def evaluate_model(self, model, test_loader, ds_name):
        """
        评估模型性能（包含详细指标和可视化）

        Args:
            model: 要评估的模型
            test_loader: 测试数据加载器
            ds_name: 数据集名称
            save_samples: 是否保存预测样本
        """
        model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_samples = 0
        logger.info("Start Evaluate Model")

        with torch.no_grad():
            for A, B, Label, _ in test_loader:
                A = A.contiguous().to(self.args.device, non_blocking=True)
                B = B.contiguous().to(self.args.device, non_blocking=True)
                Label = Label.contiguous().to(self.args.device, non_blocking=True)

                with torch.autocast(device_type=self.args.device, dtype=torch.float16):
                    pred = model(A, B)
                    loss = cross_entropy(pred[0].contiguous(), Label)

                total_loss += loss.item() * A.size(0)
                num_samples += A.size(0)

                all_preds.append(pred[0].cpu())
                all_labels.append(Label.cpu())

        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        result_dict = get_all_metrics(pred=all_preds, label=all_labels)

        if self.wandb is not None:
            prefixed_dict = {f"test/{ds_name}/{k}": v for k, v in result_dict.items()}
            self.wandb.log(prefixed_dict)

        return result_dict

    def save_model(self, model, epoch, is_best=False):
        """
        保存模型到文件
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": vars(self.args),
        }

        save_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存到: {save_path}")

        # 记录模型到wandb
        if self.wandb is not None:
            self.wandb.save(save_path, base_path=self.save_dir)

        if is_best:
            best_path = os.path.join(self.save_dir, "model_best.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已保存到: {best_path}")

            # 记录最佳模型到wandb
            if self.wandb is not None:
                self.wandb.save(best_path, base_path=self.save_dir)

    def load_model(self, checkpoint_path):
        """
        从文件加载模型
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"checkpoint文件不存在: {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.args.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        logger.info(f"已从 {checkpoint_path} 加载模型，epoch: {epoch}")

        return epoch

    def test(self):
        """
        在所有测试数据集上评估全局模型性能
        """
        logger.info("=" * 60)
        logger.info("开始测试全局模型...")
        logger.info("=" * 60)

        metrics = self.evaluate_model(
            self.model,
            self.test_loader,
            "TESTSET",
        )

    def start_train(self):
        """
        开始联邦学习训练流程
        """
        logger.info("训练配置:")
        logger.info(f"  客户端总数: {self.args.n_clients}")
        logger.info(f"  每轮参与客户端比例: {self.args.frac}")
        logger.info(f"  训练轮数: {self.args.num_epochs}")
        logger.info(f"  客户端本地训练轮数: {self.args.num_client_epoch}")
        logger.info(f"  评估间隔: 每 {self.args.eval_interval} 轮评估一次")
        logger.info(f"  使用并行训练: {getattr(self.args, 'use_parallel', True)}")

        train_losses = []

        for round_idx in range(self.args.num_epochs):
            self.current_round = round_idx
            round_start_time = time.time()

            logger.info(f"{'=' * 60}")
            logger.info(f"训练轮次: {round_idx + 1}/{self.args.num_epochs}")
            logger.info(f"{'=' * 60}")

            m = max(int(self.args.frac * self.args.n_clients), 1)
            selected_client_indices = np.random.choice(
                range(self.args.n_clients), m, replace=False
            )

            logger.info(f"本轮选中的客户端: {selected_client_indices.tolist()}")

            if self.wandb is not None and round_idx == 0:
                self.wandb.config.update(
                    {
                        "selected_clients_per_round": m,
                        "total_clients": self.args.n_clients,
                        "client_fraction": self.args.frac,
                    }
                )

            client_models = []
            client_losses = []

            use_parallel = getattr(self.args, "use_parallel", True)

            if use_parallel:
                client_models, client_losses = self.train_clients_parallel(
                    selected_client_indices
                )
            else:
                for client_idx in selected_client_indices:
                    logger.info(f"  训练客户端 {client_idx}...")

                    client_model, client_loss = self.train_client(
                        model=self.model,
                        dataloader=self.train_loader[client_idx],
                        client_idx=client_idx,
                    )

                    client_models.append(client_model.state_dict())
                    client_losses.append(client_loss)

                    logger.info(f"  客户端 {client_idx} 训练损失: {client_loss:.4f}")

                    if self.wandb is not None:
                        self.wandb.log(
                            {
                                f"train/round_{round_idx}/client_{client_idx}_loss": client_loss,
                            },
                            step=round_idx,
                        )

            updated_weights = self.average_weights(client_models)
            self.model.load_state_dict(updated_weights)

            round_avg_loss = sum(client_losses) / len(client_losses)
            train_losses.append(round_avg_loss)

            round_time = time.time() - round_start_time

            if self.wandb is not None:
                self.wandb.log(
                    {
                        "train/round_loss": round_avg_loss,
                        "train/round_time": round_time,
                        "train/clients_per_second": m / round_time,
                    },
                    step=round_idx,
                )

            logger.info(f"轮次 {round_idx + 1} 总结:")
            logger.info(f"  - 平均训练损失: {round_avg_loss:.4f}")
            logger.info(f"  - 本轮耗时: {round_time:.2f} 秒")
            logger.info(f"  - 训练速度: {m / round_time:.2f} 客户端/秒")

            if round_idx % self.args.eval_interval == 0:
                test_metrics = self.test()


def main():
    """
    主函数：启动联邦学习训练流程
    """

    import wandb
    from assgin_ds import get_fed_dataloaders_with_allocator
    from backbone.BaseTransformer import BASE_Transformer
    from utils.args import get_fed_config

    wandb.login()
    fed_config = get_fed_config()

    project_name = "change-detection-demo"

    ds_name = {
        "LEVIR": {
            "path": fed_config.datasets + "/LEVIR",
            "n_clients": 2,
            "data_ratios": [0.6, 0.4],
            "sampler_configs": [
                {"type": "random", "shuffle": True},
                {"type": "weighted", "shuffle": True, "weights": None},
            ],
        },
        "S2Looking": {
            "path": fed_config.datasets + "/S2Looking",
            "n_clients": 4,
            "data_ratios": [0.5, 0.3, 0.15, 0.05],
            "sampler_configs": [
                {"type": "random", "shuffle": True},
                {"type": "sequential"},
                {"type": "random", "shuffle": True},
                {"type": "weighted", "shuffle": True, "weights": None},
            ],
        },
        "WHUCD": {
            "path": fed_config.datasets + "/WHUCD",
            "n_clients": 2,
            "data_ratios": [0.5, 0.5],
            "sampler_configs": [
                {"type": "random", "shuffle": True},
                {"type": "sequential"},
            ],
        },
    }

    config_dict = vars(fed_config)

    with wandb.init(project=project_name, config=config_dict) as run:
        logger.info("=" * 60)
        logger.info("正在加载数据集...")
        from assgin_ds import get_fed_dataset

        train_dict, test_dict = get_fed_dataset(args=fed_config, ds_name=ds_name)

        train_loaders, test_loader, client_info = get_fed_dataloaders_with_allocator(
            train_datasets=train_dict,
            test_datasets=test_dict,
            ds_name=ds_name,
            args=fed_config,
        )

        logger.info("数据分配完成！")
        logger.info(f"总客户端数: {len(train_loaders)}")
        logger.info(f"测试数据集数: {len(test_loader)}")

        display_client_info(train_loaders, ds_name)

        tot_client = 0
        current_client_id = 0

        for ds_name, ds_info in ds_name.items():
            n_clients = ds_info["n_clients"]
            tot_client += n_clients
            current_client_id += n_clients

        logger.info("=" * 60)
        logger.info("正在初始化模型...")

        model = BASE_Transformer(
            input_nc=3,
            output_nc=2,
            token_len=4,
            resnet_stages_num=4,
            with_pos="learned",
            enc_depth=1,
            dec_depth=8,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数量: {trainable_params:,}")
        logger.info("模型初始化完成！")

        logger.info(f"客户端数量: {tot_client}")

        Trainer = FedTrain(
            args=fed_config,
            model=model,
            train_loader=train_loaders,
            test_loader=test_loader,
            n_clients=tot_client,
        )

        Trainer.start_train()
        logger.info("训练完成！")


if __name__ == "__main__":
    logger.add("logs/{time}" + ".log", rotation="50 MB", level="DEBUG")
    main()
