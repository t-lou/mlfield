from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import components.mmperc.common.debug_ploter as debug_ploter
import components.mmperc.common.loss_logger as loss_logger
from components.mmperc.losses.detection_losses import focal_loss, l1_loss, sem_loss_fn


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
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        amp_context = nullcontext()
        scaler = None

    recent_losses: deque[float] = deque(maxlen=20)
    progress = tqdm(dataloader, desc=f"{mode.title()} {epoch}/{num_epochs}", leave=False)

    logger = loss_logger.JSONLossLLogger(f"logs/{mode}_log.json")

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
            sem_pred = sem_pred[..., :H_gt, :W_gt]

            loss_hm = focal_loss(heatmap_pred, heatmap_gt)
            loss_reg = l1_loss(reg_pred, reg_gt, mask_gt)
            loss_sem = sem_loss_fn(sem_pred, sem_gt)
            loss = loss_hm + loss_reg + loss_sem

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        do_log = id_batch % log_every == 0
        is_last_batch = id_batch == len(dataloader) - 1

        # Only sync GPU->CPU (.item()) when we actually need the value:
        # for logging, for the progress bar refresh, or on the final batch
        # (so the trailing logging.info call below has a real number).
        if do_log or is_last_batch:
            current_loss = loss.item()
            last_loss_value = current_loss

            if do_log:
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

    logging.info(f"Epoch {epoch}: loss={last_loss_value:.4f}")
