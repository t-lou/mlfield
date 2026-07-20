from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.bev_utils import get_res, grid_to_xy
from common.device import get_best_device
from datasets.a2d2_dataset import A2D2Dataset, bev_collate
from model.simple_model import SimpleModel

# ================================================================
# 1. Heatmap Top-K Extraction
# ================================================================


def topk_heatmap(heatmap: torch.Tensor, K: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract top-K heatmap peaks per class.

    Args:
        heatmap: Tensor of shape (B, C, H, W)
        K: number of top peaks per class

    Returns:
        scores: (B, C, K) top-K scores
        xs:     (B, C, K) x pixel indices
        ys:     (B, C, K) y pixel indices
    """
    B, C, H, W = heatmap.shape

    # Flatten spatial dims → (B, C, H*W)
    heatmap_flat = heatmap.view(B, C, -1)

    # Top-K per class
    scores, indices = torch.topk(heatmap_flat, K, dim=-1)

    # Convert flat indices back to (x, y)
    ys = indices // W
    xs = indices % W

    return scores, xs, ys


# ================================================================
# 2. Gather Regression Values at Peak Locations
# ================================================================


def gather_regression(reg_pred: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """
    Gather regression values at the top-K heatmap peak positions.

    Args:
        reg_pred: (B, 7, H, W)
        xs:       (B, C, K)
        ys:       (B, C, K)

    Returns:
        reg_vals: (B, C, K, 7)
    """
    B, C, K = xs.shape
    _, num_reg, H, W = reg_pred.shape
    assert num_reg == 7, "Regression head must output 7 channels (dx, dy, dz, w, l, h, yaw)"

    reg_out = torch.zeros((B, C, K, 7), device=reg_pred.device, dtype=reg_pred.dtype)

    # Vectorized gather (much faster than nested Python loops)
    for b in range(B):
        for c in range(C):
            x = xs[b, c]  # (K,)
            y = ys[b, c]  # (K,)
            reg_out[b, c] = reg_pred[b, :, y, x].T  # (K, 7)

    return reg_out


# ================================================================
# 3. Decode Boxes into World Coordinates
# ================================================================


def restore_box2d(xs: torch.Tensor, ys: torch.Tensor, reg_vals: torch.Tensor):
    """
    xs:       (B, C, K) integer grid x indices
    ys:       (B, C, K) integer grid y indices
    reg_vals: (B, C, K, 6) regression values:
              [dx, dy, log_w, log_l, sin_yaw, cos_yaw]

    Returns:
        boxes: list of B lists, each containing:
               [x, y, w, l, yaw]
    """
    B, C, K = xs.shap
    res_x, res_y = get_res()

    boxes = []

    for b in range(B):
        boxes_b = []

        for c in range(C):
            for k in range(K):
                ix = int(xs[b, c, k].item())
                iy = int(ys[b, c, k].item())

                dx, dy, log_w, log_l, sin_yaw, cos_yaw = reg_vals[b, c, k].tolist()

                # 1. Grid → world cell origin
                cell_x, cell_y = grid_to_xy(ix, iy)

                # 2. Decode center
                x = cell_x + dx * res_x
                y = cell_y + dy * res_y

                # 3. Decode size
                w = math.exp(log_w)
                l_ = math.exp(log_l)

                # 4. Decode yaw
                norm = math.sqrt(sin_yaw * sin_yaw + cos_yaw * cos_yaw) + 1e-6
                yaw = math.atan2(sin_yaw / norm, cos_yaw / norm)

                boxes_b.append([x, y, w, l_, yaw])

        boxes.append(boxes_b)

    return boxes


# ================================================================
# 4. High-Level Inference Wrapper
# ================================================================


def decode_box2d(
    heatmap=torch.Tensor,
    reg=torch.Tensor,
    K: int = 50,
):
    """
    Run inference on a batch and decode bounding boxes.

    Args:
        heatmap:  Existence heatmap (B, 1, H_bev, W_bev)
        reg:      BBox parameters   (B, 6, H_bev, W_bev)
        K:        number of top-K peaks per class

    Returns:
        boxes:  list of decoded boxes per batch
        scores: (B, C, K) heatmap scores
    """

    # ---------------------------------------------------------
    # Top-K heatmap peaks
    # ---------------------------------------------------------
    scores, xs, ys = topk_heatmap(heatmap, K)

    # ---------------------------------------------------------
    # Gather regression values at peak locations
    # ---------------------------------------------------------
    reg_vals = gather_regression(reg, xs, ys)

    # ---------------------------------------------------------
    # Decode boxes into world coordinates
    # ---------------------------------------------------------
    boxes = restore_box2d(xs, ys, reg_vals)

    return boxes, scores


class ModelInferenceWrapper:
    def __init__(self, ckpt_dir="checkpoints"):
        # 1. Build model on CPU
        self.model = SimpleModel().to("cpu")

        # 2. Load checkpoint safely
        latest_path = os.path.join(ckpt_dir, "simple_model_latest.pt")
        state = torch.load(latest_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        # 3. Move model to best device
        self.device = get_best_device()
        self.model = self.model.to(self.device)

    def infer_a2d2_dataset(self, path_dataset: str, path_output: str, K: int = 50):
        assert path_output.endswith(".npz"), "path_output must be an .npz file"
        out_dir = os.path.dirname(path_output)
        os.makedirs(out_dir, exist_ok=True)

        dataset = A2D2Dataset(root=path_dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=bev_collate,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        for idx, batch in enumerate(dataloader):
            # 1. Prepare input
            points = batch["points"].to(self.device)
            images = batch["camera"].to(self.device)

            # 2. Forward pass again to get semantic logits
            with torch.no_grad():
                pred = self.model(points, images)
                boxes, scores = decode_box2d(pred["heatmap"], pred["reg"], K=K)
                sem_logits = pred["sem_logits"][0].cpu()  # (C, H, W)
                sem_pred = sem_logits.argmax(dim=0).numpy()  # (H, W)

            # # 3. Convert semantic prediction to RGB (same as debug plot)
            # class_to_color = batch["semantics_mapping_color"][0]
            # sem_rgb = np.zeros((*sem_pred.shape, 3), dtype=np.uint8)
            # for cid, rgb in class_to_color:
            #     sem_rgb[sem_pred == cid] = rgb

            # # 4. Encode semantic RGB as PNG bytes
            # png_buffer = io.BytesIO()
            # Image.fromarray(sem_rgb).save(png_buffer, format="PNG")
            # png_bytes = png_buffer.getvalue()

            # 5. Save NPZ
            np.savez_compressed(
                os.path.join(out_dir, f"sample_{idx:06d}.npz"),
                points=batch["points"][0].numpy(),
                points_timestamp=batch["points_timestamp"][0].numpy(),
                pred_boxes=np.array(boxes[0], dtype=np.float32),
                pred_scores=scores.cpu().numpy(),
                gt_boxes=batch["gt_boxes"][0],
                # sem_logits=sem_logits.numpy(),  # raw logits
                # sem_rgb=sem_rgb,  # RGB visualization
                # sem_rgb_png=png_bytes,  # PNG bytes
                sem_pred=sem_pred,  # decoded class map
                # just save the sem prediction and decode to RGB in visualizer
                semantics_mapping_color=batch["semantics_mapping_color"],
                semantics_mapping_name=batch["semantics_mapping_name"],
            )

        print(f"Saved inference results to: {out_dir}")
