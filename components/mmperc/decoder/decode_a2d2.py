from __future__ import annotations

import math
import os
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from components.dataset.a2d2_dataset import A2D2Dataset, Split, bev_collate
from components.definitions.mmperc import MmpercParams
from components.mmperc.model.simple_model import SimpleModel
from components.utils.bev_utils import get_res, grid_to_xy
from components.utils.logger import logger

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
        reg_pred: (B, 8, H, W)
        xs:       (B, C, K)
        ys:       (B, C, K)

    Returns:
        reg_vals: (B, C, K, 8)
    """
    B, C, K = xs.shape
    _, num_reg, H, W = reg_pred.shape
    assert num_reg == 8, "Regression head must output 8 channels (dx, dy, dz, log_w, log_l, log_h, sin_yaw, cos_yaw)"

    reg_out = torch.zeros((B, C, K, 8), device=reg_pred.device, dtype=reg_pred.dtype)

    # Vectorized gather (much faster than nested Python loops)
    for b in range(B):
        for c in range(C):
            x = xs[b, c]  # (K,)
            y = ys[b, c]  # (K,)
            reg_out[b, c] = reg_pred[b, :, y, x].T  # (K, 8)

    return reg_out


# ================================================================
# 3. Decode Boxes into World Coordinates
# ================================================================


def restore_box3d(xs: torch.Tensor, ys: torch.Tensor, reg_vals: torch.Tensor, params: MmpercParams):
    """
    xs:       (B, C, K) integer grid x indices
    ys:       (B, C, K) integer grid y indices
    reg_vals: (B, C, K, 8) regression values:
              [dx, dy, dz, log_w, log_l, log_h, sin_yaw, cos_yaw]

    Returns:
        boxes: list of B lists, each containing:
               [x, y, z, w, l, h, yaw]
    """
    B, C, K = xs.shape
    res_x, res_y = get_res(params)
    z_min, z_max = params.bev_params.z_range
    z_ref = (z_min + z_max) / 2.0

    boxes = []

    for b in range(B):
        boxes_b = []

        for c in range(C):
            for k in range(K):
                ix = int(xs[b, c, k].item())
                iy = int(ys[b, c, k].item())

                dx, dy, dz, log_w, log_l, log_h, sin_yaw, cos_yaw = reg_vals[b, c, k].tolist()

                # 1. Grid → world cell origin
                cell_x, cell_y = grid_to_xy(ix, iy, params)

                # 2. Decode center
                x = cell_x + dx * res_x
                y = cell_y + dy * res_y
                z = z_ref + dz

                # 3. Decode size
                w = math.exp(log_w)
                l_ = math.exp(log_l)
                h = math.exp(log_h)

                # 4. Decode yaw
                norm = math.sqrt(sin_yaw * sin_yaw + cos_yaw * cos_yaw) + 1e-6
                yaw = math.atan2(sin_yaw / norm, cos_yaw / norm)

                boxes_b.append([x, y, z, w, l_, h, yaw])

        boxes.append(boxes_b)

    return boxes


# ================================================================
# 4. High-Level Inference Wrapper
# ================================================================


def decode_box3d(
    heatmap: torch.Tensor,
    reg: torch.Tensor,
    params: MmpercParams,
    K: int = 50,
):
    """
    Run inference on a batch and decode 3D bounding boxes.

    Args:
        heatmap:  Existence heatmap (B, 1, H_bev, W_bev)
        reg:      BBox parameters   (B, 8, H_bev, W_bev)
                  [dx, dy, dz, log_w, log_l, log_h, sin_yaw, cos_yaw]
        K:        number of top-K peaks per class

    Returns:
        boxes:  list of B lists, each item [x, y, z, w, l, h, yaw]
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
    boxes = restore_box3d(xs, ys, reg_vals, params)

    return boxes, scores


class ModelInferenceWrapper:
    def __init__(self, ckpt: Path, params: MmpercParams, device: torch.device):
        self.device = device
        self.params = params

        # 1. Build model on CPU
        self.model = SimpleModel(params=params).to("cpu")

        # 2. Load checkpoint
        state = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        # 3. Move model to target device
        self.model = self.model.to(self.device)

    def infer_a2d2_dataset(self, params, path_output: str, K: int = 50):
        assert path_output.endswith(".npz"), "path_output must be an .npz file"
        out_dir = os.path.dirname(path_output)
        os.makedirs(out_dir, exist_ok=True)

        params = self.params
        dataset_eval = A2D2Dataset(path_tar=Path(params.path_data), params=params, split=Split.VAL)
        dataloader = DataLoader(
            dataset_eval,
            batch_size=4,
            shuffle=False,
            collate_fn=partial(bev_collate, params=params),
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
                boxes, scores = decode_box3d(pred["bbox_heatmap"], pred["bbox_reg"], params=params, K=K)
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

        logger.info(f"Saved inference results to: {out_dir}")
