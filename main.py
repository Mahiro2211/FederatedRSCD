import copy
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler

from assgin_ds import get_fed_dataloaders_with_allocator, get_fed_dataset
from backbone.BaseTransformer import BASE_Transformer
from configs import get_dataset_configs, get_model_config
from loss import cross_entropy
from utils.args import get_fed_config
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
        self,
        args,
        model: nn.Module,
        train_loader: list,
        test_loader,
        n_clients: int,
    ):
        """
        初始化联邦学习训练器

        Args:
            args: 训练配置参数
            model: 全局模型
            train_loader: 各客户端训练数据加载器列表 [dataloader0, dataloader1, ...]
            test_loader: 测试数据加载器
            n_clients: 客户端总数
        """
        self.args = args
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args.n_clients = n_clients
        self.current_round = 0

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

        self._setup_mixed_precision()
        self._setup_save_directory()
        self._setup_wandb()

    def _setup_mixed_precision(self):
        """配置混合精度训练"""
        self.scaler = GradScaler()

    def _setup_save_directory(self):
        """配置保存目录"""
        self.save_dir = os.path.join(
            self.args.save_dir, f"fed_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"模型和结果将保存到: {self.save_dir}")

    def _setup_wandb(self):
        """配置WandB日志"""
        try:
            import wandb

            self.wandb = wandb
            logger.info("WandB已初始化")
        except ImportError:
            self.wandb = None
            logger.warning("WandB未安装，将跳过日志记录")

        if self.wandb is not None:
            self._log_model_config_to_wandb()

    def _log_model_config_to_wandb(self):
        """记录模型配置到WandB"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.wandb.config.update(
            {
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
                "num_gpus": torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0,
                "device": self.args.device,
            }
        )

    def train_client(
        self, model: nn.Module, dataloader, client_idx: int, progress=None
    ) -> Tuple[nn.Module, float]:
        """
        在单个客户端上进行本地训练（单进程版本）

        Args:
            model: 客户端初始模型（全局模型的副本）
            dataloader: 客户端训练数据加载器
            client_idx: 客户端索引
            progress: 进度条对象（可选）

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
                f"客户端 {client_idx} - Epoch {epoch + 1}/{self.args.num_client_epoch} 完成，"
                f"损失: {avg_epoch_loss:.4f}, 耗时: {epoch_time:.2f}秒"
            )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return client_model, avg_loss

    def average_weights(
        self, clients_model: List[Dict], client_weights: Optional[List[float]] = None
    ) -> Dict:
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

        if client_weights is None:
            client_weights = [1.0 / len(clients_model)] * len(clients_model)
        else:
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        avg_weights = clients_model[0].copy()

        for key in avg_weights.keys():
            avg_weights[key] = avg_weights[key] * client_weights[0]

            for i in range(1, len(clients_model)):
                avg_weights[key] += clients_model[i][key] * client_weights[i]

        return avg_weights

    def evaluate_model(self, model: nn.Module, test_loader, ds_name: str) -> Dict:
        """
        评估模型性能（包含详细指标和可视化）

        Args:
            model: 要评估的模型
            test_loader: 测试数据加载器
            ds_name: 数据集名称

        Returns:
            dict: 评估指标字典
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

    def save_model(self, model: nn.Module, epoch: int, is_best: bool = False) -> None:
        """
        保存模型到文件

        Args:
            model: 要保存的模型
            epoch: 当前epoch
            is_best: 是否为最佳模型
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": vars(self.args),
        }

        save_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存到: {save_path}")

        if self.wandb is not None:
            self.wandb.save(save_path, base_path=self.save_dir)

        if is_best:
            best_path = os.path.join(self.save_dir, "model_best.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已保存到: {best_path}")

            if self.wandb is not None:
                self.wandb.save(best_path, base_path=self.save_dir)

    def load_model(self, checkpoint_path: str) -> int:
        """
        从文件加载模型

        Args:
            checkpoint_path: 模型检查点文件路径

        Returns:
            int: 加载的epoch号
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"checkpoint文件不存在: {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.args.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        logger.info(f"已从 {checkpoint_path} 加载模型，epoch: {epoch}")

        return epoch

    def test(self) -> Dict:
        """
        在所有测试数据集上评估全局模型性能

        Returns:
            dict: 测试指标字典
        """
        logger.info("=" * 60)
        logger.info("开始测试全局模型...")
        logger.info("=" * 60)

        metrics = self.evaluate_model(self.model, self.test_loader, "TESTSET")

        return metrics

    def start_train(self) -> None:
        """
        开始联邦学习训练流程
        """
        self._log_training_config()

        for round_idx in range(self.args.num_epochs):
            self.current_round = round_idx
            round_start_time = time.time()

            self._log_round_header(round_idx)

            m = max(int(self.args.frac * self.args.n_clients), 1)
            selected_client_indices = np.random.choice(
                range(self.args.n_clients), m, replace=False
            ).tolist()

            logger.info(f"本轮选中的客户端: {selected_client_indices}")

            self._log_round_config_to_wandb(round_idx, m)

            client_models, client_losses = self._train_clients(selected_client_indices)

            self._aggregate_and_update_model(client_models, client_losses)

            round_avg_loss = sum(client_losses) / len(client_losses)
            round_time = time.time() - round_start_time

            self._log_round_metrics_to_wandb(round_idx, round_avg_loss, round_time, m)

            self._log_round_summary(round_idx, round_avg_loss, round_time, m)

            if round_idx % self.args.eval_interval == 0:
                self.test()

    def _log_training_config(self):
        """记录训练配置"""
        logger.info("训练配置:")
        logger.info(f"  客户端总数: {self.args.n_clients}")
        logger.info(f"  每轮参与客户端比例: {self.args.frac}")
        logger.info(f"  训练轮数: {self.args.num_epochs}")
        logger.info(f"  客户端本地训练轮数: {self.args.num_client_epoch}")
        logger.info(f"  评估间隔: 每 {self.args.eval_interval} 轮评估一次")

    def _log_round_header(self, round_idx: int):
        """记录轮次标题"""
        logger.info(f"{'=' * 60}")
        logger.info(f"训练轮次: {round_idx + 1}/{self.args.num_epochs}")
        logger.info(f"{'=' * 60}")

    def _log_round_config_to_wandb(self, round_idx: int, m: int):
        """记录轮次配置到WandB"""
        if self.wandb is not None and round_idx == 0:
            self.wandb.config.update(
                {
                    "selected_clients_per_round": m,
                    "total_clients": self.args.n_clients,
                    "client_fraction": self.args.frac,
                }
            )

    def _train_clients(
        self, selected_client_indices: List[int]
    ) -> Tuple[List[Dict], List[float]]:
        """
        训练选中的客户端

        Args:
            selected_client_indices: 选中的客户端索引列表

        Returns:
            tuple: (客户端模型列表, 客户端损失列表)
        """
        return self._train_clients_sequential(selected_client_indices)

    def _train_clients_sequential(
        self, selected_client_indices: List[int]
    ) -> Tuple[List[Dict], List[float]]:
        """
        顺序训练客户端

        Args:
            selected_client_indices: 选中的客户端索引列表

        Returns:
            tuple: (客户端模型列表, 客户端损失列表)
        """
        client_models = []
        client_losses = []

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
                        f"train/round_{self.current_round}/client_{client_idx}_loss": client_loss,
                    },
                    step=self.current_round,
                )

        return client_models, client_losses

    def _aggregate_and_update_model(
        self, client_models: List[Dict], client_losses: List[float]
    ) -> None:
        """
        聚合客户端模型并更新全局模型

        Args:
            client_models: 客户端模型列表
            client_losses: 客户端损失列表
        """
        updated_weights = self.average_weights(client_models)
        self.model.load_state_dict(updated_weights)

    def _log_round_metrics_to_wandb(
        self, round_idx: int, round_avg_loss: float, round_time: float, m: int
    ) -> None:
        """
        记录轮次指标到WandB

        Args:
            round_idx: 轮次索引
            round_avg_loss: 平均损失
            round_time: 轮次耗时
            m: 参与客户端数量
        """
        if self.wandb is not None:
            self.wandb.log(
                {
                    "train/round_loss": round_avg_loss,
                    "train/round_time": round_time,
                    "train/clients_per_second": m / round_time,
                },
                step=round_idx,
            )

    def _log_round_summary(
        self, round_idx: int, round_avg_loss: float, round_time: float, m: int
    ) -> None:
        """
        记录轮次总结

        Args:
            round_idx: 轮次索引
            round_avg_loss: 平均损失
            round_time: 轮次耗时
            m: 参与客户端数量
        """
        logger.info(f"轮次 {round_idx + 1} 总结:")
        logger.info(f"  - 平均训练损失: {round_avg_loss:.4f}")
        logger.info(f"  - 本轮耗时: {round_time:.2f} 秒")
        logger.info(f"  - 训练速度: {m / round_time:.2f} 客户端/秒")


def load_model_and_config(args):
    """
    加载模型和计算总客户端数

    Args:
        args: 配置参数

    Returns:
        tuple: (模型, 总客户端数)
    """
    dataset_configs = get_dataset_configs(args.datasets)

    tot_client = sum(config["n_clients"] for config in dataset_configs.values())

    logger.info("=" * 60)
    logger.info("正在初始化模型...")

    model_config = get_model_config("BASE_Transformer")
    model = BASE_Transformer(**model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    logger.info("模型初始化完成！")

    logger.info(f"客户端数量: {tot_client}")

    return model, tot_client


def load_data(args):
    """
    加载数据集

    Args:
        args: 配置参数

    Returns:
        tuple: (训练数据加载器列表, 测试数据加载器, 客户端信息列表)
    """
    logger.info("=" * 60)
    logger.info("正在加载数据集...")

    dataset_configs = get_dataset_configs(args.datasets)

    train_dict, test_dict = get_fed_dataset(args=args, ds_name=dataset_configs)

    train_loaders, test_loader, client_info = get_fed_dataloaders_with_allocator(
        train_datasets=train_dict,
        test_datasets=test_dict,
        ds_name=dataset_configs,
        args=args,
    )

    logger.info("数据分配完成！")
    logger.info(f"总客户端数: {len(train_loaders)}")
    logger.info(f"测试数据集数: {len(test_loader)}")

    display_client_info(train_loaders, dataset_configs)

    return train_loaders, test_loader, dataset_configs


def setup_wandb(project_name: str, config_dict: dict):
    """
    设置WandB

    Args:
        project_name: 项目名称
        config_dict: 配置字典

    Returns:
        WandB run对象
    """
    import wandb

    wandb.login()
    return wandb.init(project=project_name, config=config_dict)


def main():
    """
    主函数：启动联邦学习训练流程
    """
    logger.add("logs/{time}.log", rotation="50 MB", level="DEBUG")

    fed_config = get_fed_config()
    project_name = "change-detection-demo"

    config_dict = vars(fed_config)

    with setup_wandb(project_name, config_dict) as run:
        train_loaders, test_loader, dataset_configs = load_data(fed_config)
        model, tot_client = load_model_and_config(fed_config)

        trainer = FedTrain(
            args=fed_config,
            model=model,
            train_loader=train_loaders,
            test_loader=test_loader,
            n_clients=tot_client,
        )

        trainer.start_train()
        logger.info("训练完成！")


if __name__ == "__main__":
    main()
