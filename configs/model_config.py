"""
Model configurations for change detection
"""

MODEL_CONFIGS = {
    "BASE_Transformer": {
        "input_nc": 3,
        "output_nc": 2,
        "token_len": 4,
        "resnet_stages_num": 4,
        "with_pos": "learned",
        "enc_depth": 1,
        "dec_depth": 8,
    }
}


def get_model_config(model_name: str) -> dict:
    """
    Get model configuration by name

    Args:
        model_name: Name of the model

    Returns:
        Model configuration dictionary
    """
    return MODEL_CONFIGS.get(model_name, {})
