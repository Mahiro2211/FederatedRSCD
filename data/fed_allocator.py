"""
联邦学习数据分配器
支持分层分配策略：
- 每个数据集独立分配给其客户端
- 每个数据集内部支持数量分布偏斜（Non-IID）
- 支持不同的采样器策略
"""

import torch
from torch.utils.data import DataLoader, Subset
from data.fed_sampler import SamplerFactory


class FedDataAllocator:
    """
    联邦数据分配器
    为每个数据集按客户端数量分配数据，支持固定比例的数量分布偏斜
    """

    def __init__(self, dataset_dict, ds_config):
        """
        Args:
            dataset_dict: 数据集字典 {"dataset_name": dataset}
            ds_config: 数据集配置
                {
                    "LEVIR": {
                        "n_clients": 2,
                        "data_ratios": [0.6, 0.4],  # 客户端级数量偏斜
                        "sampler_configs": [
                            {"type": "random", "shuffle": True},
                            {"type": "weighted", "shuffle": True, "weights": None}
                        ]
                    },
                    "S2Looking": {
                        "n_clients": 4,
                        "data_ratios": [0.5, 0.3, 0.15, 0.05],  # 严重偏斜
                        "sampler_configs": [...]
                    },
                    ...
                }
        """
        self.dataset_dict = dataset_dict
        self.ds_config = ds_config
        self.client_info = []

    def allocate_datasets(self):
        """
        为每个数据集分配到客户端

        Returns:
            client_datasets: 客户端数据集列表
            client_info: 客户端信息列表
        """
        client_datasets = []
        client_info = []

        current_client_id = 0

        for ds_name, ds_info in self.ds_config.items():
            if ds_name not in self.dataset_dict:
                continue

            dataset = self.dataset_dict[ds_name]
            n_clients = ds_info["n_clients"]
            data_ratios = ds_info.get("data_ratios", None)
            sampler_configs = ds_info.get("sampler_configs", [{}] * n_clients)

            if data_ratios is None:
                data_ratios = [1.0 / n_clients] * n_clients

            data_subsets = self._allocate_data_to_clients(dataset, data_ratios)

            for i in range(n_clients):
                client_datasets.append(data_subsets[i])
                client_info.append(
                    {
                        "client_id": current_client_id + i,
                        "dataset_name": ds_name,
                        "sampler_config": sampler_configs[i]
                        if i < len(sampler_configs)
                        else {},
                    }
                )

            current_client_id += n_clients

        self.client_info = client_info
        return client_datasets, client_info

    def _allocate_data_to_clients(self, dataset, data_ratios):
        """
        将数据集按比例分配给客户端

        Args:
            dataset: 原始数据集
            data_ratios: 分配比例列表

        Returns:
            data_subsets: 客户端数据子集列表
        """
        dataset_size = len(dataset)
        data_subsets = []
        start_idx = 0

        for ratio in data_ratios:
            num_samples = int(dataset_size * ratio)
            end_idx = start_idx + num_samples

            indices = list(range(start_idx, min(end_idx, dataset_size)))
            data_subsets.append(Subset(dataset, indices))

            start_idx = end_idx

        return data_subsets

    def create_dataloaders(self, batch_size=8, num_workers=4, shuffle=None):
        """
        为每个客户端创建 DataLoader，使用不同的采样器

        Args:
            batch_size: 批大小
            num_workers: 工作进程数
            shuffle: 是否打乱（会被采样器配置覆盖）

        Returns:
            train_loaders: 训练数据加载器列表
            test_loaders: 测试数据加载器列表（如果提供测试集）
        """
        client_datasets, client_info = self.allocate_datasets()

        train_loaders = []

        for i, (client_dataset, info) in enumerate(zip(client_datasets, client_info)):
            sampler_config = info["sampler_config"]

            sampler = SamplerFactory.create_sampler(client_dataset, sampler_config)

            dataloader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_loaders.append(dataloader)

        return train_loaders, self.client_info


def get_fed_dataloaders(train_datasets, test_datasets, ds_name, args):
    """
    便捷函数：创建联邦学习数据加载器

    Args:
        train_datasets: 训练数据集字典
        test_datasets: 测试数据集字典
        ds_name: 数据集配置
        args: 训练参数

    Returns:
        train_loaders: 训练数据加载器列表（分配给各客户端）
        test_loaders: 测试数据加载器列表（每个数据集一个完整测试加载器）
        client_info: 客户端信息列表
    """
    train_allocator = FedDataAllocator(train_datasets, ds_name)

    train_loaders, client_info = train_allocator.create_dataloaders(
        batch_size=args.batch_size, num_workers=4
    )

    # 测试数据不分配，直接为每个数据集创建完整的测试加载器
    test_loaders = []
    for ds_name, ds_info in ds_name.items():
        if ds_name not in test_datasets:
            continue

        test_dataset = test_datasets[ds_name]
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_loaders.append(test_loader)

    return train_loaders, test_loaders, client_info
