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
    batchA, batchB, batchL, name = [], [], [], []
    for index in range(len(batch)):
        batchA.append(batch[index]["A"])
        batchB.append(batch[index]["B"])
        batchL.append(batch[index]["L"])
        name.append(batch[index]["name"])
    batchA = torch.stack(batchA)
    batchB = torch.stack(batchB)
    batchL = torch.stack(batchL)
    return batchA, batchB, batchL, name
