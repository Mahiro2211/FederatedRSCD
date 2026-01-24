"""
Docstring for data.collate_func:
用于客制化编写一些处理数据的代码
"""

import torch


def collate_func(batch):
    """
    Docstring for collate_func

    :param batch: 遍历dataset时候返回的batch是字典里头包含着name, A, B, label这些数据
    :return: 处理后每个类型的数据都会被处理成一个batchsize的数据返回
    """
    if not batch:
        return {
            "A": torch.tensor([]),
            "B": torch.tensor([]),
            "L": torch.tensor([]),
            "name": [],
        }

    # Validate that first item has required keys
    if not isinstance(batch[0], dict) or not all(
        key in batch[0] for key in ["A", "B", "L", "name"]
    ):
        raise ValueError(
            "Each item in batch must be a dictionary with keys 'A', 'B', 'L', 'name'"
        )

    batchA = torch.stack([item["A"] for item in batch])
    batchB = torch.stack([item["B"] for item in batch])
    batchL = torch.stack([item["L"] for item in batch])
    names = [item["name"] for item in batch]

    return batchA, batchB, batchL, names
