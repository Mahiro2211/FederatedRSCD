import argparse


def get_fed_config():
    parser = argparse.ArgumentParser(description="Federated Learning Configuration")

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_client_epoch",
        type=int,
        default=2,
        help="Number of training epoch for each indenpendent client",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluation interval"
    )
    parser.add_argument(
        "--multi_losses",
        action="store_true",
        default=False,
        help="Whether to use multiple losses",
    )
    parser.add_argument(
        "--loss_type", type=str, default="ce", help="Type of loss function to use"
    )
    parser.add_argument(
        "--multi_scale_infer",
        action="store_true",
        default=False,
        help="Whether to use multi-scale inference",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        default=False,
        help="Whether to save results",
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of classes for training"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay for optimization",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon value for Adam/AdamW optimizers",
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.9, 0.999],
        help="Beta values for Adam/AdamW optimizers",
    )

    # Federated learning specific arguments
    parser.add_argument(
        "--niid",
        action="store_true",
        default=True,
        help="Whether to use non-IID data distribution",
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=20,
        help="Number of shards for data partitioning",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.5,
        help="Fraction of clients to participate in each round",
    )

    # Model arguments
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension of the model"
    )
    parser.add_argument(
        "--num_classes_model",
        type=int,
        default=10,
        help="Number of output classes in model",
    )
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")

    # Data arguments
    parser.add_argument(
        "--datasets",
        type=str,
        default="/home/dhm/dataset/",
        help="root directory of datasets may contain multiple datasets",
    )

    # Performance optimization arguments
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_models",
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--num_workers_dataloader",
        type=int,
        default=4,
        help="Number of workers for data loading (per client)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = get_fed_config()
    print(config)
