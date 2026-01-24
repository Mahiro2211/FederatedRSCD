"""
联邦学习采样器工厂
支持创建不同类型的采样器
"""

from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
import numpy as np


class SamplerFactory:
    """采样器工厂类"""

    @staticmethod
    def create_sampler(dataset, config):
        """
        创建采样器

        Args:
            dataset: PyTorch 数据集
            config: 采样器配置字典
                {
                    "type": "random" | "sequential" | "weighted",
                    "weights": list/None  # 仅对 weighted 有效
                }

        Returns:
            sampler: PyTorch 采样器
        """
        sampler_type = config.get("type", "random")

        if sampler_type == "random":
            return RandomSampler(dataset)

        elif sampler_type == "sequential":
            return SequentialSampler(dataset)

        elif sampler_type == "weighted":
            weights = config.get("weights")
            if weights is None:
                num_samples = len(dataset)
                weights = np.ones(num_samples)
            else:
                weights = np.array(weights)
            return WeightedRandomSampler(
                weights=weights, num_samples=len(dataset), replacement=True
            )

        else:
            raise ValueError(f"Unsupported sampler type: {sampler_type}")
