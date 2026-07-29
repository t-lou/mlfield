from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import components.mmperc.common.debug_ploter as debug_ploter
import components.mmperc.common.loss_logger as loss_logger
from components.mmperc.losses.detection_losses import (
    focal_loss,
    l1_loss,
    semantic_ce_loss,
    semantic_invalid_aux_loss,
)


def _build_semantic_class_weights(
    sem_gt: torch.Tensor,
    num_classes: int,
    invalid_class_id: int,
    invalid_scale: float,
    w_min: float,
    w_max: float,
) -> torch.Tensor:
    """
    Build class-balanced CE weights from the current batch histogram.
    Invalid class stays supervised but can be downscaled to avoid dominance.
    """
    counts = torch.bincount(sem_gt.view(-1), minlength=num_classes).float()
    freq = counts / counts.sum().clamp_min(1.0)

    weights = 1.0 / torch.sqrt(freq + 1e-6)
    weights = weights / weights.mean().clamp_min(1e-6)
    weights = weights.clamp(min=w_min, max=w_max)
    weights[invalid_class_id] = weights[invalid_class_id] * invalid_scale
    return weights.to(device=sem_gt.device, dtype=torch.float32)


# ================================================================
# Train or Eval Epoch
# ================================================================
def run_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int = 0,
    num_epochs: int = 1,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    train: bool = True,
    log_every: int = 10,
    debug_plot_every: int = 200,
):
    if train:
        model.train()
        mode = "train"
    else:
        model.eval()
        mode = "eval"

    if mode == "train" and device.type == "cuda":
        amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        amp_context = nullcontext()
        scaler = None

    recent_losses: deque[float] = deque(maxlen=20)
    progress = tqdm(dataloader, desc=f"{mode.title()} {epoch}/{num_epochs}", leave=False)

    logger = loss_logger.JSONLossLLogger(f"logs/{mode}_log.json")

    params = getattr(model, "_params", None)
    num_sem_classes = params.num_sem_classes
    invalid_class_id = num_sem_classes - 1
    weight_sem_loss = params.weight_sem_loss
    weight_loss_hm = params.weight_loss_hm
    weight_loss_reg = params.weight_loss_reg
    sem_invalid_ce_weight = params.sem_invalid_ce_weight
    sem_use_class_balanced_ce = params.sem_use_class_balanced_ce
    sem_ce_weight_min = params.sem_ce_weight_min
    sem_ce_weight_max = params.sem_ce_weight_max
    sem_invalid_aux_weight = params.sem_invalid_aux_weight
    sem_invalid_bce_pos_weight = params.sem_invalid_bce_pos_weight

    invalid_tp = 0
    invalid_fp = 0
    invalid_fn = 0
    invalid_gt_pixels = 0
    invalid_pred_pixels = 0
    total_sem_pixels = 0

    # keep a running sum on-device; only synced to CPU when we actually log
    last_loss_value: float = 0.0

    assert dataloader.pin_memory, "pin_memory=False — non_blocking=True transfers won't overlap with compute"

    for id_batch, batch in enumerate(progress):
        # non_blocking transfers only help if the DataLoader uses pin_memory=True;
        # if it doesn't yet, set pin_memory=True on the DataLoader for this to matter.
        points: torch.Tensor = batch["points"].to(device, non_blocking=True)
        images: torch.Tensor = batch["camera"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        fwd_context = amp_context if train else (amp_context if device.type == "cuda" else nullcontext())
        no_grad_context = nullcontext() if train else torch.no_grad()

        with no_grad_context, fwd_context:
            pred: Dict[str, torch.Tensor] = model(points, images)
            heatmap_pred = pred["bbox_heatmap"]
            reg_pred = pred["bbox_reg"]
            sem_pred = pred["sem_logits"]

            heatmap_gt = batch["heatmap_gt"].to(device, non_blocking=True)
            reg_gt = batch["reg_gt"].to(device, non_blocking=True)
            mask_gt = batch["mask_gt"].to(device, non_blocking=True)
            sem_gt = batch["semantics"].to(device, non_blocking=True)

            H_gt, W_gt = sem_gt.shape[-2], sem_gt.shape[-1]
            if sem_pred.shape[-2:] != (H_gt, W_gt):
                sem_pred = F.interpolate(sem_pred, size=(H_gt, W_gt), mode="bilinear", align_corners=False)

            class_weights = None
            if sem_use_class_balanced_ce:
                class_weights = _build_semantic_class_weights(
                    sem_gt=sem_gt,
                    num_classes=num_sem_classes,
                    invalid_class_id=invalid_class_id,
                    invalid_scale=sem_invalid_ce_weight,
                    w_min=sem_ce_weight_min,
                    w_max=sem_ce_weight_max,
                )

            loss_hm = focal_loss(heatmap_pred, heatmap_gt)
            loss_reg = l1_loss(reg_pred, reg_gt, mask_gt)
            # Semantic loss is a combination of weighted CE and auxiliary invalid BCE. The reason for the auxiliary
            # BCE is that the invalid class is often dominant, and the CE loss can be dominated by it. The auxiliary
            # BCE helps the model learn to separate invalid pixels from valid ones.
            loss_sem_ce = semantic_ce_loss(sem_pred, sem_gt, class_weights=class_weights)
            loss_sem_invalid = semantic_invalid_aux_loss(
                logits=sem_pred,
                target=sem_gt,
                invalid_class_id=invalid_class_id,
                pos_weight=sem_invalid_bce_pos_weight,
            )
            loss_sem = loss_sem_ce + sem_invalid_aux_weight * loss_sem_invalid
            loss = weight_loss_hm * loss_hm + weight_loss_reg * loss_reg + weight_sem_loss * loss_sem
            sum_weights = weight_loss_hm + weight_loss_reg + weight_sem_loss
            loss = loss / sum_weights

            sem_pred_class = sem_pred.argmax(dim=1)
            gt_invalid = sem_gt == invalid_class_id
            pred_invalid = sem_pred_class == invalid_class_id
            invalid_tp += (gt_invalid & pred_invalid).sum().item()
            invalid_fp += ((~gt_invalid) & pred_invalid).sum().item()
            invalid_fn += (gt_invalid & (~pred_invalid)).sum().item()
            invalid_gt_pixels += gt_invalid.sum().item()
            invalid_pred_pixels += pred_invalid.sum().item()
            total_sem_pixels += sem_gt.numel()

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        do_log = id_batch % log_every == 0
        is_last_batch = id_batch == len(dataloader) - 1

        # Only sync GPU->CPU (.item()) when we actually need the value:
        # for logging, for the progress bar refresh, or on the final batch
        # (so the trailing logging.info call below has a real number).
        if do_log or is_last_batch or mode == "eval":
            current_loss = loss.item()
            last_loss_value = current_loss

            if do_log or mode == "eval":
                record = loss_logger.EpochLoss(
                    epoch_id=epoch,
                    batch_id=id_batch,
                    loss_hm=loss_hm.detach().item(),
                    loss_reg=loss_reg.detach().item(),
                    loss_sem=loss_sem.detach().item(),
                    loss_total=current_loss,
                )
                logger.append(record)

                recent_losses.append(current_loss)
                avg20 = sum(recent_losses) / len(recent_losses)
                progress.set_postfix(loss=f"{current_loss:.2f}", avg20=f"{avg20:.2f}")

        # debug plots (unchanged cadence, already infrequent)
        if id_batch % debug_plot_every == 0:
            debug_ploter.export_bbox_heatmap_debug(heatmap_pred[0, 0], heatmap_gt[0, 0], epoch, id_batch)
            class_to_color = batch["semantics_mapping_color"][0]
            debug_ploter.export_semantic_debug(sem_pred[0], sem_gt[0], class_to_color, epoch, id_batch)

    # Step scheduler after epoch
    if train and scheduler is not None:
        scheduler.step()

    if mode == "eval" and total_sem_pixels > 0:
        invalid_precision = invalid_tp / max(invalid_tp + invalid_fp, 1)
        invalid_recall = invalid_tp / max(invalid_tp + invalid_fn, 1)
        invalid_iou = invalid_tp / max(invalid_tp + invalid_fp + invalid_fn, 1)
        gt_invalid_ratio = invalid_gt_pixels / total_sem_pixels
        pred_invalid_ratio = invalid_pred_pixels / total_sem_pixels

        logging.info(
            "Eval semantic invalid metrics | "
            f"iou={invalid_iou:.4f}, precision={invalid_precision:.4f}, recall={invalid_recall:.4f}, "
            f"gt_ratio={gt_invalid_ratio:.4f}, pred_ratio={pred_invalid_ratio:.4f}"
        )

    logging.info(f"Epoch {epoch}: loss={last_loss_value:.4f}")
