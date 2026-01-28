from __future__ import annotations

import io
import math
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import common.params as params
from common.bev_utils import get_res, grid_to_xy_stride
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


def decode_boxes(xs: torch.Tensor, ys: torch.Tensor, reg_vals: torch.Tensor):
    """
    xs:       (B, C, K) integer grid x indices
    ys:       (B, C, K) integer grid y indices
    reg_vals: (B, C, K, 6) regression values:
              [dx, dy, log_w, log_l, sin_yaw, cos_yaw]

    Returns:
        boxes: list of B lists, each containing:
               [x, y, z, w, l, h, yaw]
    """
    B, C, K = xs.shape
    stride = params.BACKBONE_STRIDE
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
                cell_x, cell_y = grid_to_xy_stride(ix, iy)

                # 2. Decode center
                x = cell_x + dx * (res_x * stride)
                y = cell_y + dy * (res_y * stride)

                # 3. Decode size
                w = math.exp(log_w)
                l_ = math.exp(log_l)

                # 4. Decode yaw
                yaw = math.atan2(sin_yaw, cos_yaw)

                # 5. z and h are NOT encoded — set defaults or predict elsewhere
                z = 0.0
                h = 1.5

                boxes_b.append([x, y, z, w, l_, h, yaw])

        boxes.append(boxes_b)

    return boxes


# ================================================================
# 4. High-Level Inference Wrapper
# ================================================================


def model_inference(
    model: torch.nn.Module,
    points: torch.Tensor,
    images: torch.Tensor,
    K: int = 50,
):
    """
    Run inference on a batch and decode bounding boxes.

    Args:
        model:  detection model
        points: (B, N, 4) lidar input
        images: (B, 3, H, W) camera input
        K:      number of top-K peaks per class

    Returns:
        boxes:  list of decoded boxes per batch
        scores: (B, C, K) heatmap scores
    """
    model.eval()

    with torch.no_grad():
        # ---------------------------------------------------------
        # 1. Forward pass through your full multimodal model
        # ---------------------------------------------------------
        pred = model(points, images)

        heatmap = pred["heatmap"]  # (B, C, H_bev, W_bev)
        reg = pred["reg"]  # (B, 6, H_bev, W_bev)

        # ---------------------------------------------------------
        # 2. Top-K heatmap peaks
        # ---------------------------------------------------------
        scores, xs, ys = topk_heatmap(heatmap, K)

        # ---------------------------------------------------------
        # 3. Gather regression values at peak locations
        # ---------------------------------------------------------
        reg_vals = gather_regression(reg, xs, ys)

        # ---------------------------------------------------------
        # 4. Decode boxes into world coordinates
        # ---------------------------------------------------------
        boxes = decode_boxes(xs, ys, reg_vals)

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

    def infer(self, points: torch.Tensor, images: torch.Tensor, K: int = 50):
        # Ensure inputs are on the same device as the model
        points = points.to(self.device)
        images = images.to(self.device)

        return model_inference(self.model, points, images, K)

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
            points = batch["points"].to(self.device)
            images = batch["camera"].to(self.device)

            # 1. Run inference
            boxes, scores = self.infer(points, images, K)

            # 2. Forward pass again to get semantic logits
            with torch.no_grad():
                pred = self.model(points, images)
                sem_logits = pred["sem_logits"][0].cpu()  # (C, H, W)
                sem_pred = sem_logits.argmax(dim=0).numpy()  # (H, W)

            # 3. Convert semantic prediction to RGB (same as debug plot)
            class_to_color = batch["semantics_mapping_color"][0]
            sem_rgb = np.zeros((*sem_pred.shape, 3), dtype=np.uint8)
            for cid, rgb in class_to_color:
                sem_rgb[sem_pred == cid] = rgb

            # 4. Encode semantic RGB as PNG bytes
            png_buffer = io.BytesIO()
            Image.fromarray(sem_rgb).save(png_buffer, format="PNG")
            png_bytes = png_buffer.getvalue()

            # 5. Save NPZ
            np.savez_compressed(
                os.path.join(out_dir, f"sample_{idx:06d}.npz"),
                points=batch["points"][0].numpy(),  # original point cloud
                bboxes=np.array(boxes[0], dtype=np.float32),
                heatmap_scores=scores.cpu().numpy(),
                sem_logits=sem_logits.numpy(),  # raw logits
                sem_pred=sem_pred,  # decoded class map
                sem_rgb=sem_rgb,  # RGB visualization
                sem_rgb_png=png_bytes,  # PNG bytes
            )

        print(f"Saved inference results to: {out_dir}")
