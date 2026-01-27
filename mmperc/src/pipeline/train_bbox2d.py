from __future__ import annotations

import logging
import os
import shutil
from collections import deque
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import common.debug_ploter as debug_ploter
import common.loss_logger as loss_logger
from common.archive import (
    archive_existing_model,
    ensure_dir,
    find_latest_epoch_checkpoint,
    save_checkpoint,
)
from losses.detection_losses import focal_loss, l1_loss, sem_loss_fn

# ================================================================
# Continuous Training Loop
# ================================================================


def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 1,
    ckpt_dir="checkpoints",
) -> float:
    debug_ploter.init_plot()
    ensure_dir(ckpt_dir)
    latest_path = os.path.join(ckpt_dir, "simple_model_latest.pt")

    # ---------------------------------------------------------
    # 1. Detect previous epoch checkpoints (continuous training)
    # ---------------------------------------------------------
    resume_info = find_latest_epoch_checkpoint(ckpt_dir)

    if resume_info is not None:
        last_epoch, last_ckpt_path = resume_info
        logging.info(f"[Resume] Found previous checkpoint: {last_ckpt_path}")

        # Load model weights
        state = torch.load(last_ckpt_path, map_location="cpu")
        model.load_state_dict(state)

        shutil.copy2(last_ckpt_path, latest_path)
        logging.info(f"[Resume] Copied previous checkpoint to latest: {latest_path}")

        start_epoch = last_epoch + 1

    else:
        # No epoch checkpoints → archive existing latest if any
        archive_existing_model(latest_path)
        start_epoch = 0

    # ---------------------------------------------------------
    # 2. Training loop with autosave + crash recovery
    # ---------------------------------------------------------
    try:
        for epoch in range(start_epoch, start_epoch + num_epochs):
            loss = train_one_epoch(
                model, dataloader, optimizer, device, epoch=epoch, num_epochs=start_epoch + num_epochs
            )
            logging.info(f"Epoch {epoch}: loss={loss:.4f}")

            # Save epoch checkpoint
            epoch_path = os.path.join(ckpt_dir, f"simple_model_epoch_{epoch:04d}.pt")
            save_checkpoint(model, epoch_path)

            shutil.copy2(epoch_path, latest_path)
            logging.info(f"[Latest] Updated latest checkpoint → {latest_path}")

    except Exception as e:
        # Crash recovery
        crash_path = os.path.join(ckpt_dir, "simple_model_crash.pt")
        save_checkpoint(model, crash_path)

        shutil.copy2(crash_path, latest_path)
        logging.info("[Crash] Saved crash checkpoint and updated latest.")

        raise e


# ================================================================
# One Epoch
# ================================================================


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    num_epochs: int = 1,
    ckpt_dir="checkpoints",
) -> float:
    model.train()

    recent_losses: deque[float] = deque(maxlen=20)
    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

    logger = loss_logger.JSONLossLLogger("logs/train_log.json")

    id_batch = 0
    for batch in progress:
        points: torch.Tensor = batch["points"].to(device)
        images: torch.Tensor = batch["camera"].to(device)
        # gt_boxes directly from dataloader

        optimizer.zero_grad()

        pred: Dict[str, torch.Tensor] = model(points, images)
        heatmap_pred = pred["bbox_heatmap"]
        reg_pred = pred["bbox_reg"]
        sem_pred = pred["sem_logits"]

        heatmap_gt: torch.Tensor = batch["heatmap_gt"].to(device).clone()
        reg_gt: torch.Tensor = batch["reg_gt"].to(device).clone()
        mask_gt: torch.Tensor = batch["mask_gt"].to(device).clone()
        sem_gt: torch.Tensor = batch["semantics"].to(device).clone()

        # refactor later
        H_gt, W_gt = sem_gt.shape[-2], sem_gt.shape[-1]
        sem_pred = sem_pred[..., :H_gt, :W_gt]
        sem_loss = sem_loss_fn(sem_pred, sem_gt)
        loss_hm = focal_loss(heatmap_pred, heatmap_gt)
        loss_reg = l1_loss(reg_pred, reg_gt, mask_gt)
        loss = loss_hm + loss_reg + sem_loss

        loss_record = loss_logger.EpochLoss(
            epoch_id=epoch,
            batch_id=id_batch,
            loss_hm=loss_hm,
            loss_reg=loss_reg,
            loss_sem=sem_loss,
            loss_total=loss,
        )
        logger.append(loss_record)

        loss.backward()
        optimizer.step()

        # Plot for debugging
        debug_ploter.export_bbox_heatmap_debug(heatmap_pred[0, 0, ...], heatmap_gt[0, 0, ...], epoch, id_batch)
        class_to_color = batch["semantics_mapping_color"][0]  # from your npz
        debug_ploter.export_semantic_debug(
            sem_pred[0, ...],  # (C, H, W)
            sem_gt[0, ...],  # (H, W)
            class_to_color,  # list of (cid, rgb)
            epoch,
            id_batch,
        )

        current_loss = loss.item()
        recent_losses.append(current_loss)
        avg20 = sum(recent_losses) / len(recent_losses)

        progress.set_postfix(loss=f"{current_loss:.2f}", avg20=f"{avg20:.2f}")

        id_batch += 1

    return sum(recent_losses) / len(recent_losses)
