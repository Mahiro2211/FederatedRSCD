import numpy as np
from loguru import logger
from rich.console import Console

console = Console()


def display_client_info(train_loaders, ds_name_config):
    """
    显示所有客户端的训练样本和采样模式信息

    Args:
        train_loaders: 训练数据加载器列表
        ds_name_config: 数据集配置字典
    """
    console.print("\n[bold yellow]客户端信息一览[/bold yellow]")
    console.print("=" * 80)

    client_id = 0
    total_samples = 0

    for ds_name, ds_info in ds_name_config.items():
        n_clients = ds_info["n_clients"]
        data_ratios = ds_info["data_ratios"]
        sampler_configs = ds_info["sampler_configs"]

        console.print(f"\n[bold cyan]数据集: {ds_name}[/bold cyan]")
        console.print(f"  路径: {ds_info['path']}")
        console.print(f"  客户端总数: {n_clients}")

        for i in range(n_clients):
            # 计算该客户端的样本数量
            client_idx = client_id + i
            if client_idx < len(train_loaders):
                dataloader = train_loaders[client_idx]
                try:
                    # 尝试获取数据集大小
                    dataset_size = len(dataloader.dataset)
                    client_samples = int(dataset_size * data_ratios[i])
                except:
                    # 如果无法获取数据集大小，使用估计值
                    client_samples = f"比例: {data_ratios[i] * 100:.1f}%"

                console.print(f"\n  [yellow]客户端 {client_id + i + 1}:[/yellow]")
                console.print(f"    - 数据集: {ds_name}")
                console.print(f"    - 训练样本数: {client_samples}")
                console.print(f"    - 采样模式: {sampler_configs[i]['type']}")
                console.print(
                    f"    - 是否随机打乱: {sampler_configs[i].get('shuffle', False)}"
                )

                if (
                    sampler_configs[i]["type"] == "weighted"
                    and sampler_configs[i].get("weights") is not None
                ):
                    console.print("    - 权重配置: 自定义权重")
                elif sampler_configs[i]["type"] == "weighted":
                    console.print("    - 权重配置: 默认权重")

                if isinstance(client_samples, int):
                    total_samples += client_samples

        client_id += n_clients

    console.print("\n[bold green]总结:[/bold green]")
    console.print(f"  - 总客户端数: {len(train_loaders)}")
    console.print(f"  - 总训练样本数: {total_samples}")
    console.print("=" * 80)


def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(["iou_" + str(i) for i in range(n_class)], iu))

    cls_precision = dict(
        zip(["precision_" + str(i) for i in range(n_class)], precision)
    )
    cls_recall = dict(zip(["recall_" + str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(["F1_" + str(i) for i in range(n_class)], F1))

    score_dict = {"acc": acc, "miou": mean_iu, "mf1": mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""

    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(
            num_classes * label_gt[mask].astype(int) + label_pred[mask],
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict["miou"]


def get_all_metrics(pred, label):
    cm = get_confuse_matrix(
        2,
        label.detach().cpu().numpy(),
        np.argmax(pred.detach().cpu().numpy(), axis=1),
    )
    result_dict = cm2score(cm)
    logger.info(
        f"acc: {result_dict['acc']}, miou: {result_dict['miou']}, mf1: {result_dict['mf1']}"
    )
    logger.info(
        f"iou_0: {result_dict['iou_0']}, F1_0: {result_dict['F1_0']}, F1_1: {result_dict['F1_1']}"
    )
    logger.info(
        f"recall_0: {result_dict['recall_0']}, recall_1: {result_dict['recall_1']}"
    )
    logger.info(
        f"precision_0: {result_dict['precision_0']}, precision_1: {result_dict['precision_1']}"
    )
    return result_dict
