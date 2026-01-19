from __future__ import annotations

from collections import deque
from typing import Dict

import torch
from label.bev_labels import generate_bev_labels_bbox2d
from losses.detection_losses import focal_loss, l1_loss
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    num_epochs: int = 1,
) -> float:
    """
    Train the model for a single epoch.

    Args:
        model:      The neural network being trained.
        dataloader: PyTorch DataLoader yielding batches of A2D2 data.
        optimizer:  Optimizer used for gradient updates.
        device:     Target device (CPU or GPU).
        epoch:      Current epoch number (for logging).
        num_epochs: Total number of epochs (for logging).

    Returns:
        float: Average loss over the last 20 batches (sliding window).
    """

    model.train()

    # Sliding window of recent losses (smooths progress bar)
    recent_losses: deque[float] = deque(maxlen=20)

    # tqdm progress bar for batch-level feedback
    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

    for batch in progress:
        # Move tensors to device
        points: torch.Tensor = batch["points"].to(device)
        images: torch.Tensor = batch["camera"].to(device)
        gt_boxes = batch["gt_boxes"]  # stays on CPU for label generation

        optimizer.zero_grad()

        # -----------------------------
        # Forward pass
        # -----------------------------
        pred: Dict[str, torch.Tensor] = model(points, images)
        heatmap_pred = pred["heatmap"]
        reg_pred = pred["reg"]

        # -----------------------------
        # Generate BEV supervision
        # -----------------------------
        heatmap_gt, reg_gt, mask_gt = generate_bev_labels_bbox2d(gt_boxes)
        heatmap_gt = heatmap_gt.to(device)
        reg_gt = reg_gt.to(device)
        mask_gt = mask_gt.to(device)

        # -----------------------------
        # Compute losses
        # -----------------------------
        loss_hm = focal_loss(heatmap_pred, heatmap_gt)
        loss_reg = l1_loss(reg_pred, reg_gt, mask_gt)
        loss = loss_hm + loss_reg

        # -----------------------------
        # Backpropagation
        # -----------------------------
        loss.backward()
        optimizer.step()

        # -----------------------------
        # Sliding-window average loss
        # -----------------------------
        current_loss = loss.item()
        recent_losses.append(current_loss)
        avg20 = sum(recent_losses) / len(recent_losses)

        # Update progress bar display
        progress.set_postfix(loss=f"{current_loss:.2f}", avg20=f"{avg20:.2f}")

    # Return the smoothed loss for logging
    return sum(recent_losses) / len(recent_losses)
