import copy
import os
import time
from datetime import datetime
from tqdm import tqdm
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler

from loss import cross_entropy
from train import train_client_worker
from utils.tools import display_client_info
from utils.tools import get_all_metrics

# Rich控制台
console = Console()


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

        # 创建梯度缩放器用于混合精度训练
        self.scaler = GradScaler()

        # 创建保存模型和结果的目录
        self.save_dir = os.path.join(
            args.save_dir, f"fed_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"模型和结果将保存到: {self.save_dir}")

        # 初始化wandb（如果可用）
        try:
            import wandb

            self.wandb = wandb
            # 尝试获取当前的wandb run
            if wandb.run is not None:
                self.wandb_run = wandb.run
            else:
                self.wandb_run = None
            logger.info("WandB已初始化")
        except ImportError:
            self.wandb = None
            self.wandb_run = None
            logger.warning("WandB未安装，将跳过日志记录")

    
        # 记录模型参数到wandb
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

        显示详细的训练进度，包括epoch级别和batch级别的进度

        Args:
            model: 客户端初始模型（全局模型的副本）
            dataloader: 客户端训练数据加载器
            client_idx: 客户端索引
            progress: Rich Progress对象（可选）

        Returns:
            tuple: (训练后的模型, 平均损失)
        """
        # 深拷贝模型，避免影响全局模型
        client_model = copy.deepcopy(model)
        client_model.train()

        # 创建优化器
        optimizer = torch.optim.Adam(
            client_model.parameters(),
            lr=self.args.lr,
            betas=self.args.betas,
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
        )

        # 创建梯度缩放器
        client_scaler = GradScaler()

        total_loss = 0.0
        num_batches = 0
        total_batches = len(dataloader) * self.args.num_client_epoch

        # 在客户端上进行多个epoch的本地训练
        for epoch in range(self.args.num_client_epoch):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_task = None

            # 创建epoch级别的进度条
            if progress is not None:
                epoch_task = progress.add_task(
                    f"[cyan]客户端 {client_idx} - Epoch {epoch + 1}/{self.args.num_client_epoch}",
                    total=len(dataloader),
                )
                iterator = dataloader
            else:
                # 使用tqdm
                from tqdm import tqdm

                iterator = tqdm(
                    dataloader,
                    desc=f"客户端 {client_idx} - Epoch {epoch + 1}/{self.args.num_client_epoch}",
                )

            for batch_idx, (A, B, Label, _) in enumerate(iterator):
                A = A.contiguous().to(self.args.device, non_blocking=True)
                B = B.contiguous().to(self.args.device, non_blocking=True)
                Label = Label.contiguous().to(self.args.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # 使用自动混合精度训练（AMP）
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

                # 更新Rich进度条
                if progress is not None and epoch_task is not None:
                    progress.update(
                        epoch_task,
                        advance=1,
                        description=f"[cyan]客户端 {client_idx} - Epoch {epoch + 1}/{self.args.num_client_epoch} - Loss: {loss.item():.4f}",
                    )

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0

            logger.info(
                f"    客户端 {client_idx} - Epoch {epoch + 1}/{self.args.num_client_epoch} 完成，损失: {avg_epoch_loss:.4f}, 耗时: {epoch_time:.2f}秒"
            )

        # 计算平均损失
        avg_loss = total_loss / (num_batches * self.args.num_client_epoch)

        return client_model, avg_loss

    def train_clients_parallel(self, selected_client_indices, progress=None):
        """
        使用多进程并行训练多个客户端

        多进程并行可以显著提高训练速度，特别是在客户端数量较多时

        Args:
            selected_client_indices: 选中的客户端索引列表
            progress: Rich Progress对象（可选）

        Returns:
            tuple: (客户端模型状态字典列表, 客户端损失列表)
        """
        import multiprocessing as mp

        # 获取全局模型的状态字典
        global_state_dict = self.model.state_dict()

        # 准备每个客户端的参数
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

        # 确定使用的进程数
        n_workers = min(
            self.args.n_workers if hasattr(self.args, "n_workers") else cpu_count(),
            len(selected_client_indices),
        )
        logger.info(
            f"使用 {n_workers} 个进程并行训练 {len(selected_client_indices)} 个客户端"
        )

        # 使用进程池并行训练客户端
        # 注意：在Linux/WSL上使用CUDA需要使用'spawn' start method
        ctx = mp.get_context("spawn")
        client_models = []
        client_losses = []

        with ctx.Pool(processes=n_workers) as pool:
            # 如果有进度条对象，使用它；否则使用tqdm
            if progress is not None:
                # 使用外部进度条
                task = progress.add_task(
                    "[cyan]并行训练客户端中...", total=len(client_args)
                )
                results = []
                for result in pool.imap(train_client_worker, client_args):
                    results.append(result)
                    progress.update(task, advance=1)
            else:
                # 使用tqdm（兼容性更好）
                from tqdm import tqdm

                results = list(
                    tqdm(
                        pool.imap(train_client_worker, client_args),
                        total=len(client_args),
                        desc="并行训练客户端",
                    )
                )

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

    def evaluate_model(
        self, model, test_loader, ds_name, save_samples=True, progress=None
    ):
        """
        评估模型性能（包含详细指标和可视化）

        Args:
            model: 要评估的模型
            test_loader: 测试数据加载器
            ds_name: 数据集名称
            save_samples: 是否保存预测样本
            progress: Rich Progress对象（可选）
        """
        model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_samples = 0

        inference_times = []

        with torch.no_grad():
            for A, B, Label, _ in tqdm(test_loader, total=len(test_loader)):
                A = A.contiguous().to(self.args.device, non_blocking=True)
                B = B.contiguous().to(self.args.device, non_blocking=True)
                Label = Label.contiguous().to(self.args.device, non_blocking=True)

                start_time = time.time()

                with torch.autocast(device_type=self.args.device, dtype=torch.float16):
                    pred = model(A, B)
                    loss = cross_entropy(pred[0].contiguous(), Label)

                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                total_loss += loss.item() * A.size(0)
                num_samples += A.size(0)

                all_preds.append(pred[0].cpu())
                all_labels.append(Label.cpu())

        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_labels= torch.cat(all_labels, dim=0).cpu()

        result_dict = get_all_metrics(pred=all_preds, label=all_labels)
        # 计算推理速度

        # 记录测试指标到wandb
        if self.wandb is not None:
            prefix = f"test/{ds_name}"
            self.wandb.log(
               result_dict 
            )

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

    def test(self, progress=None):
        """
        在所有测试数据集上评估全局模型性能

        Args:
            progress: Rich Progress对象（可选）
        """
        logger.info("=" * 60)
        logger.info("开始测试全局模型...")
        logger.info("=" * 60)

        metrics = self.evaluate_model(
            self.model,
            self.test_loader,
            "TESTSET",
            save_samples=True,
            progress=progress,
        )


    def start_train(self):
        """
        开始联邦学习训练流程
        """
        # 使用Rich显示训练配置
        console.print("\n[bold blue]训练配置[/bold blue]")
        console.print(f"  客户端总数: [cyan]{self.args.n_clients}[/cyan]")
        console.print(f"  每轮参与客户端比例: [cyan]{self.args.frac}[/cyan]")
        console.print(f"  训练轮数: [cyan]{self.args.num_epochs}[/cyan]")
        console.print(
            f"  客户端本地训练轮数: [cyan]{self.args.num_client_epoch}[/cyan]"
        )
        console.print(
            f"  评估间隔: [cyan]每 {self.args.eval_interval} 轮评估一次[/cyan]"
        )
        console.print(
            f"  使用并行训练: [cyan]{getattr(self.args, 'use_parallel', True)}[/cyan]"
        )

        train_losses = []
        best_f1 = 0.0

        # 使用Rich进度条显示整体训练进度
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # 创建总体训练任务
            overall_task = progress.add_task(
                "[bold green]联邦学习训练进度", total=self.args.num_epochs
            )

            for round_idx in range(self.args.num_epochs):
                self.current_round = round_idx  # 用于wandb日志记录
                round_start_time = time.time()

                progress.console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
                progress.console.print(
                    f"[bold cyan]训练轮次: {round_idx + 1}/{self.args.num_epochs}[/bold cyan]"
                )
                progress.console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

                # 随机选择参与本轮训练的客户端
                m = max(int(self.args.frac * self.args.n_clients), 1)
                selected_client_indices = np.random.choice(
                    range(self.args.n_clients), m, replace=False
                )

                logger.info(f"本轮选中的客户端: {selected_client_indices.tolist()}")

                # 记录训练配置到wandb
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
                        selected_client_indices, progress
                    )
                else:
                    for client_idx in selected_client_indices:
                        logger.info(f"  训练客户端 {client_idx}...")

                        client_model, client_loss = self.train_client(
                            model=self.model,
                            dataloader=self.train_loader[client_idx],
                            client_idx=client_idx,
                            progress=progress,
                        )

                        client_models.append(client_model.state_dict())
                        client_losses.append(client_loss)

                        logger.info(
                            f"  客户端 {client_idx} 训练损失: {client_loss:.4f}"
                        )

                        # 记录客户端损失到wandb
                        if self.wandb is not None:
                            self.wandb.log(
                                {
                                    f"train/round_{round_idx}/client_{client_idx}_loss": client_loss,
                                },
                                step=round_idx,
                            )

                # 聚合客户端模型参数
                updated_weights = self.average_weights(client_models)
                self.model.load_state_dict(updated_weights)

                # 计算本轮平均损失
                round_avg_loss = sum(client_losses) / len(client_losses)
                train_losses.append(round_avg_loss)

                round_time = time.time() - round_start_time

                # 记录轮次级别指标到wandb
                if self.wandb is not None:
                    import wandb

                    self.wandb.log(
                        {
                            "train/round_loss": round_avg_loss,
                            "train/round_time": round_time,
                            "train/clients_per_second": m / round_time,
                            "train/selected_clients": selected_client_indices.tolist(),
                        },
                        step=round_idx,
                    )

                # 使用Rich显示本轮训练结果
                progress.console.print(
                    f"\n[bold yellow]轮次 {round_idx + 1} 总结:[/bold yellow]"
                )
                progress.console.print(
                    f"  - 平均训练损失: [red]{round_avg_loss:.4f}[/red]"
                )
                progress.console.print(
                    f"  - 本轮耗时: [cyan]{round_time:.2f}[/cyan] 秒"
                )
                progress.console.print(
                    f"  - 训练速度: [cyan]{m / round_time:.2f}[/cyan] 客户端/秒"
                )

                # 更新总体进度
                progress.update(overall_task, advance=1)

                # 定期评估模型
                if round_idx % self.args.eval_interval == 0:

                    test_metrics = self.test(progress=progress)



def main():
    """
    主函数：启动联邦学习训练流程
    """
    from datetime import datetime

    from loguru import logger

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

    if __name__ == "__main__":
        # 将配置转换为字典格式
        config_dict = vars(fed_config)

        with wandb.init(project=project_name, config=config_dict) as run:

            print(f"\n{'=' * 60}")

            # ========== 第1步：加载数据集 ==========
            console.print("[bold blue]正在加载数据集...[/bold blue]")
            from assgin_ds import get_fed_dataset

            train_dict, test_dict = get_fed_dataset(args=fed_config, ds_name=ds_name)

            train_loaders, test_loader, client_info = (
                get_fed_dataloaders_with_allocator(
                    train_datasets=train_dict,
                    test_datasets=test_dict,
                    ds_name=ds_name,
                    args=fed_config,
                )
            )

            console.print("\n[bold green]✅ 数据分配完成！[/bold green]")
            console.print(f"总客户端数: [cyan]{len(train_loaders)}[/cyan]")
            console.print(f"测试数据集数: [cyan]{len(test_loader)}[/cyan]")

            # 显示所有客户端的训练样本和采样模式信息
            display_client_info(train_loaders, ds_name)

            tot_client = 0
            current_client_id = 0

            for ds_name, ds_info in ds_name.items():
                n_clients = ds_info["n_clients"]
                tot_client += n_clients
                current_client_id += n_clients

            # ========== 第2步：初始化模型 ==========
            console.print("\n[bold blue]正在初始化模型...[/bold blue]")

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
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            console.print(f"  - 总参数量: [cyan]{total_params:,}[/cyan]")
            console.print(f"  - 可训练参数量: [cyan]{trainable_params:,}[/cyan]")
            console.print("[bold green]✅ 模型初始化完成！[/bold green]\n")

            # ========== 第3步：启动联邦学习训练 ==========
            logger.info(f"客户端数量: {tot_client}")

            Trainer = FedTrain(
                args=fed_config,
                model=model,
                train_loader=train_loaders,
                test_loader=test_loader,
                n_clients=tot_client,
            )

            Trainer.start_train()
            console.print("\n[bold green]🎉 训练完成！[/bold green]")


if __name__ == "__main__":
    logger.add('logs/{time}'+ '.log',
            rotation='50 MB', level='DEBUG')
    main()
