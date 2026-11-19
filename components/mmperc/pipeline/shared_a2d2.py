from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext

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
def run_one_epoch(  # noqa: C901
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int = 0,
    num_epochs: int = 1,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    train: bool = True,
    log_every: int = 10,
    debug_plot_every: int = 0,
):
    if train:
        model.train()
        mode = "train"
    else:
        model.eval()
        mode = "eval"

    supported_device_type: str | None = device.type if device.type in ("cuda", "mps") else None
    amp_context = (
        torch.autocast(device_type=supported_device_type, dtype=torch.float16)
        if supported_device_type is not None
        else nullcontext()
    )
    scaler = (
        torch.amp.GradScaler(supported_device_type, enabled=True)
        if train and supported_device_type is not None
        else None
    )

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

    invalid_counts = torch.zeros(5, dtype=torch.int64, device=device) if not train else None
    total_sem_pixels = 0

    last_loss_value: float = 0.0

    non_blocking = device.type == "mps" or (device.type == "cuda" and dataloader.pin_memory)

    for id_batch, batch in enumerate(progress):
        points: torch.Tensor = batch["points"].to(device, non_blocking=non_blocking)
        images: torch.Tensor = batch["camera"].to(device, non_blocking=non_blocking)

        if train:
            optimizer.zero_grad(set_to_none=True)

        no_grad_context = nullcontext() if train else torch.no_grad()

        with no_grad_context, amp_context:
            pred: dict[str, torch.Tensor] = model(points, images)
            heatmap_pred = pred["bbox_heatmap"]
            reg_pred = pred["bbox_reg"]
            sem_pred = pred["sem_logits"]

            heatmap_gt = batch["heatmap_gt"].to(device, non_blocking=non_blocking)
            reg_gt = batch["reg_gt"].to(device, non_blocking=non_blocking)
            mask_gt = batch["mask_gt"].to(device, non_blocking=non_blocking)
            sem_gt = batch["semantics"].to(device, non_blocking=non_blocking)

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

            if invalid_counts is not None:
                gt_invalid = sem_gt == invalid_class_id
                pred_invalid = sem_pred.argmax(dim=1) == invalid_class_id
                invalid_counts += torch.stack(
                    (
                        (gt_invalid & pred_invalid).sum(),
                        ((~gt_invalid) & pred_invalid).sum(),
                        (gt_invalid & (~pred_invalid)).sum(),
                        gt_invalid.sum(),
                        pred_invalid.sum(),
                    )
                )
                total_sem_pixels += sem_gt.numel()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        do_log = log_every > 0 and id_batch % log_every == 0
        is_last_batch = id_batch == len(dataloader) - 1

        if do_log:
            loss_values = (
                torch.stack(
                    (
                        loss_hm.detach().float(),
                        loss_reg.detach().float(),
                        loss_sem.detach().float(),
                        loss.detach().float(),
                    )
                )
                .cpu()
                .tolist()
            )
            loss_hm_value, loss_reg_value, loss_sem_value, current_loss = loss_values
            last_loss_value = current_loss

            record = loss_logger.EpochLoss(
                epoch_id=epoch,
                batch_id=id_batch,
                loss_hm=loss_hm_value,
                loss_reg=loss_reg_value,
                loss_sem=loss_sem_value,
                loss_total=current_loss,
            )
            logger.append(record)

            recent_losses.append(current_loss)
            avg20 = sum(recent_losses) / len(recent_losses)
            progress.set_postfix(loss=f"{current_loss:.2f}", avg20=f"{avg20:.2f}")
        elif is_last_batch:
            last_loss_value = float(loss.detach().item())

        if debug_plot_every > 0 and id_batch % debug_plot_every == 0:
            debug_ploter.export_bbox_heatmap_debug(heatmap_pred[0, 0], heatmap_gt[0, 0], epoch, id_batch)
            class_to_color = batch["semantics_mapping_color"][0]
            debug_ploter.export_semantic_debug(sem_pred[0], sem_gt[0], class_to_color, epoch, id_batch)

    logger.close()

    # Step scheduler after epoch
    if train and scheduler is not None:
        scheduler.step()

    if invalid_counts is not None and total_sem_pixels > 0:
        invalid_tp, invalid_fp, invalid_fn, invalid_gt_pixels, invalid_pred_pixels = invalid_counts.cpu().tolist()
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
