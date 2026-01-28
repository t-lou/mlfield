from __future__ import annotations

import logging
from collections import deque
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import common.debug_ploter as debug_ploter
import common.loss_logger as loss_logger
from losses.detection_losses import focal_loss, l1_loss, sem_loss_fn


# ================================================================
# Train or f Epoch
# ================================================================
def run_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int = 0,
    num_epochs: int = 1,
    optimizer: torch.optim.Optimizer | None = None,
    train: bool = True,
):
    if train:
        model.train()
        mode = "train"
    else:
        model.eval()
        mode = "eval"

    recent_losses: deque[float] = deque(maxlen=20)
    progress = tqdm(dataloader, desc=f"{mode.title()} {epoch}/{num_epochs}", leave=False)

    logger = loss_logger.JSONLossLLogger(f"logs/{mode}_log.json")

    id_batch = 0

    for batch in progress:
        points: torch.Tensor = batch["points"].to(device)
        images: torch.Tensor = batch["camera"].to(device)

        if train:
            optimizer.zero_grad()

        # forward
        pred: Dict[str, torch.Tensor] = model(points, images)
        heatmap_pred = pred["bbox_heatmap"]
        reg_pred = pred["bbox_reg"]
        sem_pred = pred["sem_logits"]

        # ground truth
        heatmap_gt = batch["heatmap_gt"].to(device)
        reg_gt = batch["reg_gt"].to(device)
        mask_gt = batch["mask_gt"].to(device)
        sem_gt = batch["semantics"].to(device)

        # semantic resize
        H_gt, W_gt = sem_gt.shape[-2], sem_gt.shape[-1]
        sem_pred = sem_pred[..., :H_gt, :W_gt]

        # compute losses
        loss_hm = focal_loss(heatmap_pred, heatmap_gt)
        loss_reg = l1_loss(reg_pred, reg_gt, mask_gt)
        loss_sem = sem_loss_fn(sem_pred, sem_gt)
        loss = loss_hm + loss_reg + loss_sem

        # log
        record = loss_logger.EpochLoss(
            epoch_id=epoch,
            batch_id=id_batch,
            loss_hm=loss_hm.detach().cpu().item(),
            loss_reg=loss_reg.detach().cpu().item(),
            loss_sem=loss_sem.detach().cpu().item(),
            loss_total=loss.detach().cpu().item(),
        )
        logger.append(record)

        # backward only in training
        if train:
            loss.backward()
            optimizer.step()

        # debug plots
        debug_ploter.export_bbox_heatmap_debug(heatmap_pred[0, 0], heatmap_gt[0, 0], epoch, id_batch)
        class_to_color = batch["semantics_mapping_color"][0]
        debug_ploter.export_semantic_debug(sem_pred[0], sem_gt[0], class_to_color, epoch, id_batch)

        # progress bar
        current_loss = loss.item()
        recent_losses.append(current_loss)
        avg20 = sum(recent_losses) / len(recent_losses)
        progress.set_postfix(loss=f"{current_loss:.2f}", avg20=f"{avg20:.2f}")

        id_batch += 1

    logging.info(f"Epoch {epoch}: loss={loss:.4f}")
