"""
Dataset configurations for federated learning
"""

DATASET_CONFIGS = {
    "LEVIR": {
        "path_suffix": "/LEVIR",
        "n_clients": 2,
        "data_ratios": [0.6, 0.4],
        "sampler_configs": [
            {"type": "random", "shuffle": True},
            {"type": "weighted", "shuffle": True, "weights": None},
        ],
    },
    "S2Looking": {
        "path_suffix": "/S2Looking",
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
        "path_suffix": "/WHUCD",
        "n_clients": 2,
        "data_ratios": [0.5, 0.5],
        "sampler_configs": [
            {"type": "random", "shuffle": True},
            {"type": "sequential"},
        ],
    },
}


def get_dataset_configs(base_path: str) -> dict:
    """
    Get dataset configurations with full paths

    Args:
        base_path: Base directory for datasets

    Returns:
        Dataset configurations with full paths
    """
    configs = {}
    for name, config in DATASET_CONFIGS.items():
        configs[name] = config.copy()
        configs[name]["path"] = base_path + config["path_suffix"]
    return configs
