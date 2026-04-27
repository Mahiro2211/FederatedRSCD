import torch
import torch.nn.functional as F


def _align_logits(logits, target):
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if logits.shape[-1] != target.shape[-1]:
        logits = F.interpolate(
            logits, size=target.shape[1:], mode="bilinear", align_corners=True
        ).contiguous()
    return logits, target


def cross_entropy(logits, target, weight=None, reduction="mean", ignore_index=255):
    logits, target = _align_logits(logits, target)
    return F.cross_entropy(
        input=logits,
        target=target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )


def focal_loss(logits, target, alpha=0.25, gamma=2.0, ignore_index=255):
    """
    Focal Loss for addressing class imbalance.

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    logits, target = _align_logits(logits, target)
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    pt = torch.exp(-ce)
    focal = alpha * (1 - pt) ** gamma * ce

    mask = target != ignore_index
    return focal[mask].mean()


def dice_loss(logits, target, smooth=1.0, ignore_index=255):
    """
    Dice Loss for segmentation.
    Uses soft dice computed per-sample then averaged.
    """
    logits, target = _align_logits(logits, target)
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)

    mask = target != ignore_index
    total = torch.tensor(0.0, device=logits.device)

    for c in range(num_classes):
        pred_c = probs[:, c]
        target_c = (target == c).float()
        valid = mask.float()
        intersection = (pred_c * target_c * valid).sum(dim=(1, 2))
        union = (pred_c * valid).sum(dim=(1, 2)) + (target_c * valid).sum(dim=(1, 2))
        dice_c = (2.0 * intersection + smooth) / (union + smooth)
        total = total + (1.0 - dice_c).mean()

    return total / num_classes


def ce_dice(logits, target, dice_weight=0.5, ignore_index=255):
    """
    Combined CE + Dice loss.
    """
    ce = cross_entropy(logits, target, ignore_index=ignore_index)
    dl = dice_loss(logits, target, ignore_index=ignore_index)
    return ce * (1 - dice_weight) + dl * dice_weight


LOSS_REGISTRY = {
    "ce": cross_entropy,
    "focal": focal_loss,
    "dice": dice_loss,
    "ce_dice": ce_dice,
}


def get_loss_fn(loss_type: str):
    """
    Get loss function by name.

    Args:
        loss_type: one of 'ce', 'focal', 'dice', 'ce_dice'

    Returns:
        callable loss function
    """
    fn = LOSS_REGISTRY.get(loss_type)
    if fn is None:
        raise ValueError(
            f"Loss '{loss_type}' not found. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return fn
