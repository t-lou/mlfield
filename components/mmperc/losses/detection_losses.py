import torch
import torch.nn.functional as F

from components.utils.logger import logger


def focal_loss(pred, gt, alpha=2.0, beta=4.0):
    """
    pred: (B, 1, H, W) after sigmoid
    gt:   (B, 1, H, W) with Gaussian peaks
    """
    assert pred.shape == gt.shape, f"pred shape {pred.shape} != gt shape {gt.shape}"

    # debug grid indexing
    logger.debug("GT max at:", torch.nonzero(gt[0, 0] == gt[0, 0].max()))
    logger.debug("Pred max at:", torch.nonzero(pred[0, 0] == pred[0, 0].max()))

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    # negative weights grow as gt approaches 1
    neg_weights = torch.pow(1 - gt, beta)

    # log(p)
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    if num_pos == 0:
        return neg_loss.sum()

    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def l1_loss(pred, gt, mask):
    """
    pred: (B, C, H, W)
    gt:   (B, C, H, W)
    mask: (B, 1, H, W)
    """
    mask = mask.expand_as(pred).float()
    loss = F.l1_loss(pred * mask, gt * mask, reduction="sum")
    denom = mask.sum() + 1e-6
    return loss / denom


def semantic_ce_loss(
    logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Multi-class semantic segmentation loss, it uses weighted cross-entropy to handle class imbalance.

    logits: (B, C, H, W)
    target: (B, H, W)
    class_weights: (C,) or None
    """
    return F.cross_entropy(logits, target.long(), weight=class_weights)


def semantic_invalid_aux_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    invalid_class_id: int,
    pos_weight: float = 2.0,
) -> torch.Tensor:
    """
    Binary invalid-vs-valid auxiliary loss on top of the invalid-class logit, to help the model learn to separate
    invalid pixels from valid ones.

    logits: (B, C, H, W)
    target: (B, H, W)
    invalid_class_id: int, the class ID of the invalid class
    pos_weight: float, the positive weight for BCE loss, to upweight the invalid class in the loss.
    """
    invalid_logit = logits[:, invalid_class_id, :, :]
    invalid_target = (target == invalid_class_id).float()
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(invalid_logit, invalid_target, pos_weight=pw)
