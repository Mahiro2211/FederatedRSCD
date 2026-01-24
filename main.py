from datetime import datetime

import wandb
import torch

from backbone.BaseTransformer import BASE_Transformer
from utils.args import get_fed_config
from assgin_ds import get_fed_dataloaders_with_allocator

wandb.login()


project_name = "change-detection-demo"

# 需要用到的数据集配置
ds_name = {
    "LEVIR": {
        "path": "/home/dhm/dataset/LEVIR",
        "n_clients": 2,
        "data_ratios": [0.6, 0.4],
        "sampler_configs": [
            {"type": "random", "shuffle": True},
            {"type": "weighted", "shuffle": True, "weights": None},
        ],
    },
    "S2Looking": {
        "path": "/home/dhm/dataset/S2Looking",
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
        "path": "/home/dhm/dataset/WHUCD",
        "n_clients": 2,
        "data_ratios": [0.5, 0.5],
        "sampler_configs": [
            {"type": "random", "shuffle": True},
            {"type": "sequential"},
        ],
    },
}


if __name__ == "__main__":
    fed_config = get_fed_config()

    with wandb.init(project=project_name, config=fed_config) as run:
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d_%H%M")

        print(f"\n{'=' * 60}")
        print(f"联邦学习数据分配器示例")
        print(f"{'=' * 60}\n")

        # 首先加载数据集
        print("正在加载数据集...")
        from assgin_ds import get_fed_dataset

        train_dict, test_dict = get_fed_dataset(args=fed_config, ds_name=ds_name)

        # 创建联邦学习数据加载器
        train_loaders, test_loaders, client_info = get_fed_dataloaders_with_allocator(
            train_datasets=train_dict,
            test_datasets=test_dict,
            ds_name=ds_name,
            args=fed_config,
        )

        print(f"\n{'=' * 60}")
        print(f"数据分配结果")
        print(f"{'=' * 60}")
        print(f"总客户端数: {len(train_loaders)}")
        print(f"测试数据集数: {len(test_loaders)}")
        print(f"\n客户端详情:")

        current_client_id = 0
        for ds_name, ds_info in ds_name.items():
            n_clients = ds_info["n_clients"]
            data_ratios = ds_info.get("data_ratios", None)

            print(f"\n{ds_name} 数据集:")
            print(f"  - 客户端数量: {n_clients}")
            if data_ratios:
                print(f"  - 数据分配比例: {data_ratios}")

            for i in range(n_clients):
                info = client_info[current_client_id + i]
                sampler_config = info["sampler_config"]
                print(f"    客户端 {current_client_id + i}:")
                print(f"      采样器类型: {sampler_config.get('type', 'random')}")
                print(
                    f"      训练数据量: {len(train_loaders[current_client_id + i].dataset)}"
                )

            current_client_id += n_clients

        print(f"\n测试数据集详情:")
        for i, test_loader in enumerate(test_loaders):
            print(f"  测试集 {i}: {len(test_loader.dataset)} 条数据")

        print(f"\n{'=' * 60}")
        print(f"✅ 数据分配完成！")
        print(f"  - 训练数据：每个客户端使用分配的数据子集（Non-IID）")
        print(f"  - 测试数据：使用完整的测试数据集评估全局模型")
        print(f"{'=' * 60}\n")
