"""
Model configurations and factory for change detection

Each model has a config dict (constructor kwargs) and a builder function.
The builder handles lazy imports so unused models don't occupy memory.
"""

MODEL_CONFIGS = {
    "BASE_Transformer": {
        "input_nc": 3,
        "output_nc": 2,
        "token_len": 4,
        "resnet_stages_num": 4,
        "with_pos": "learned",
        "enc_depth": 1,
        "dec_depth": 1,
    },
    "BASE_Transformer_s4_dd8": {
        "input_nc": 3,
        "output_nc": 2,
        "token_len": 4,
        "resnet_stages_num": 4,
        "with_pos": "learned",
        "enc_depth": 1,
        "dec_depth": 8,
    },
    "BASE_Transformer_s4_dd8_dedim8": {
        "input_nc": 3,
        "output_nc": 2,
        "token_len": 4,
        "resnet_stages_num": 4,
        "with_pos": "learned",
        "enc_depth": 1,
        "dec_depth": 8,
        "decoder_dim_head": 8,
    },
    "SiamUnet_diff": {},
    "SiamUnet_conc": {},
    "Unet": {},
    "ChangeFormerV1": {
        "input_nc": 3,
        "output_nc": 2,
    },
    "ChangeFormerV2": {
        "input_nc": 3,
        "output_nc": 2,
    },
    "ChangeFormerV3": {
        "input_nc": 3,
        "output_nc": 2,
    },
    "ChangeFormerV4": {
        "input_nc": 3,
        "output_nc": 2,
    },
    "ChangeFormerV5": {
        "input_nc": 3,
        "output_nc": 2,
    },
    "ChangeFormerV6": {
        "input_nc": 3,
        "output_nc": 2,
    },
    "DTCDSCN": {},
}


def get_model_config(model_name: str) -> dict:
    config = MODEL_CONFIGS.get(model_name)
    if config is None:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(MODEL_CONFIGS.keys())}"
        )
    return config


def create_model(model_name: str, args):
    """
    Factory: build a model instance by name.

    Args:
        model_name: one of MODEL_CONFIGS keys
        args: parsed CLI args (used for embed_dim etc.)

    Returns:
        nn.Module
    """
    config = get_model_config(model_name)

    if model_name.startswith("BASE_Transformer"):
        from backbone.BaseTransformer import BASE_Transformer

        return BASE_Transformer(**config)

    if model_name == "SiamUnet_diff":
        from backbone.SiamUnet_diff import SiamUnet_diff

        return SiamUnet_diff(input_nbr=3, label_nbr=2)

    if model_name == "SiamUnet_conc":
        from backbone.SiamUnet_conc import SiamUnet_conc

        return SiamUnet_conc(input_nbr=3, label_nbr=2)

    if model_name == "Unet":
        from backbone.Unet import Unet

        return Unet(input_nbr=3, label_nbr=2)

    if model_name == "DTCDSCN":
        from backbone.DTCDSCN import CDNet34

        return CDNet34(in_channels=3)

    if model_name in (
        "ChangeFormerV1",
        "ChangeFormerV2",
        "ChangeFormerV3",
        "ChangeFormerV4",
    ):
        import importlib

        cls = getattr(importlib.import_module("backbone.ChangeFormer"), model_name)
        return cls(**config)

    if model_name in ("ChangeFormerV5", "ChangeFormerV6"):
        import importlib

        cls = getattr(importlib.import_module("backbone.ChangeFormer"), model_name)
        return cls(**config, embed_dim=args.embed_dim)

    raise ValueError(f"Unknown model: {model_name}")
