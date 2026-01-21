import logging

import torch
import torch.nn.functional as F
from common.debug_ploter import DebugPloter

G_DEBUG_PLOTER = DebugPloter()


def focal_loss(pred, gt, alpha=2.0, beta=4.0):
    """
    pred: (B, 1, H, W) after sigmoid
    gt:   (B, 1, H, W) with Gaussian peaks
    """
    G_DEBUG_PLOTER.export_heatmap_debug(pred, gt)
    assert pred.shape == gt.shape, f"pred shape {pred.shape} != gt shape {gt.shape}"

    # debug grid indexing
    logging.debug("GT max at:", torch.nonzero(gt[0, 0] == gt[0, 0].max()))
    logging.debug("Pred max at:", torch.nonzero(pred[0, 0] == pred[0, 0].max()))

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
