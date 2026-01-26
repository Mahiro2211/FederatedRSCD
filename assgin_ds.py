"""
Docstring for assgin_ds
划分联邦学习数据集
"""

from loguru import logger

from data.data_processing import CDDataset


def get_fed_dataset(args, ds_name: dict):
    """
    Docstring for get_fed_dataset

    :param args: 模型超参数
    :param ds_name: 需要联邦学习的数据集信息
    :type ds_name: dict
    :return dict {"key" : torch.util.data.Dataset}
    """
    ds_trian_dict = {name: None for name in ds_name.keys()}
    ds_test_dict = {name: None for name in ds_name.keys()}
    for name, info in ds_name.items():
        ds_trian_dict[name] = CDDataset(
            root_dir=info["path"],
            split="train",
            img_size=args.img_size,
            label_transform="norm",
        )
        ds_test_dict[name] = CDDataset(
            root_dir=info["path"],
            split="test",
            img_size=args.img_size,
            label_transform="norm",
        )
        logger.info(f"{name} train size: {len(ds_trian_dict[name])}")
        logger.info(f"{name} test size: {len(ds_test_dict[name])}")
    return ds_trian_dict, ds_test_dict


def get_fed_dataloaders_with_allocator(
    train_datasets, test_datasets, ds_name: dict, args
):
    """
    使用联邦数据分配器创建 DataLoader

    Args:
        train_datasets: 训练数据集字典
        test_datasets: 测试数据集字典
        ds_name: 数据集配置（包含分配比例和采样器配置）
        args: 训练参数

    Returns:
        train_loaders: 训练数据加载器列表
        test_loaders: 测试数据加载器列表
        client_info: 客户端信息列表
    """
    from data.fed_allocator import get_fed_dataloaders

    train_loaders, test_loaders, client_info = get_fed_dataloaders(
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        ds_name=ds_name,
        args=args,
    )

    logger.info(f"Total clients: {len(train_loaders)}")
    logger.info(f"Client info: {client_info}")

    return train_loaders, test_loaders, client_info
