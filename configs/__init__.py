"""
Configuration package
"""

from .dataset_config import DATASET_CONFIGS, get_dataset_configs
from .model_config import MODEL_CONFIGS, get_model_config

__all__ = [
    "DATASET_CONFIGS",
    "get_dataset_configs",
    "MODEL_CONFIGS",
    "get_model_config",
]
