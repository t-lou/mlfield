from __future__ import annotations

import logging
import os
import shutil

import torch
from torch.utils.data import DataLoader

import common.debug_ploter as debug_ploter
from common.archive import (
    archive_existing_model,
    ensure_dir,
    find_latest_epoch_checkpoint,
    save_checkpoint,
)
from pipeline.shared_a2d2 import run_one_epoch

# ================================================================
# Continuous Training Loop
# ================================================================


def train_model(
    model: torch.nn.Module,
    dataloader_train: DataLoader,
    dataloader_eval: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 1,
    ckpt_dir="checkpoints",
):
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
            train_one_epoch(
                model=model,
                dataloader=dataloader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=start_epoch + num_epochs,
            )

            # Save epoch checkpoint
            epoch_path = os.path.join(ckpt_dir, f"simple_model_epoch_{epoch:04d}.pt")
            save_checkpoint(model, epoch_path)

            shutil.copy2(epoch_path, latest_path)
            logging.info(f"[Latest] Updated latest checkpoint → {latest_path}")

            # Evaluate the last epoch
            evaluate_one_epoch(
                model=model,
                dataloader=dataloader_eval,
                device=device,
                epoch=epoch,
                num_epochs=start_epoch + num_epochs,
            )

    except Exception as e:
        # Crash recovery
        crash_path = os.path.join(ckpt_dir, "simple_model_crash.pt")
        save_checkpoint(model, crash_path)

        shutil.copy2(crash_path, latest_path)
        logging.info("[Crash] Saved crash checkpoint and updated latest.")

        raise e


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epoch: int = 0,
    num_epochs: int = 1,
):
    run_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        num_epochs=num_epochs,
        train=True,
    )


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int = 0,
    num_epochs: int = 1,
):
    run_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=None,
        device=device,
        epoch=epoch,
        num_epochs=num_epochs,
        train=False,
    )
