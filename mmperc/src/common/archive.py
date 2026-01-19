import os
import re
import shutil
from datetime import datetime

import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_latest_epoch_checkpoint(ckpt_dir: str) -> tuple[int, str] | None:
    """
    Scan checkpoint directory for files like:
        simple_model_epoch_0005.pt
    Return (epoch_number, full_path) of the highest epoch.
    """
    if not os.path.exists(ckpt_dir):
        return None

    pattern = re.compile(r"simple_model_epoch_(\d+)\.pt")
    max_epoch = -1
    best_path = None

    for fname in os.listdir(ckpt_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_path = os.path.join(ckpt_dir, fname)

    if best_path is None:
        return None

    return max_epoch, best_path


def save_checkpoint(model, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"[Checkpoint] Saved: {path}")


def archive_existing_model(path: str):
    """
    Archive an existing model file before training starts.
    """
    if os.path.exists(path):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_dir = os.path.join(os.path.dirname(path), "archive")
        ensure_dir(archive_dir)

        base = os.path.basename(path)
        archived_path = os.path.join(archive_dir, f"{timestamp}_{base}")

        shutil.copy2(path, archived_path)
        print(f"[Archive] Previous model archived at: {archived_path}")
