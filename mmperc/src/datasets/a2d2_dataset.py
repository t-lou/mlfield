import io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import common.params as params
import numpy as np
import torch
from common.utils import rescale_image

# local dependencies
from label.bev_labels import generate_bev_labels_bbox2d
from PIL import Image
from torch.utils.data import Dataset


def bev_collate(batch):
    """
    Collate function for A2D2Dataset with fixed-size padded NPZ data.

    Lidar timestamps are ignored in this function.

    Input batch: list of dicts, each containing:
        points:    (num_lidar_points, C)
        camera:    (3, H, W)
        semantics: (H, W)
        gt_boxes:  (num_gt_boxes, 7)

    Output:
        points:      (B, num_lidar_points, C)
        camera:      (B, 3, H, W)
        semantics:   (B, H, W)
        gt_boxes:    (B, num_gt_boxes, 7)
        heatmap_gt:  (B, 1, H_bev, W_bev)
        reg_gt:      (B, 6, H_bev, W_bev)
        mask_gt:     (B, 1, H_bev, W_bev)
    """

    # Fixed-size tensors → stack directly
    points = torch.stack([item["points"] for item in batch])  # (B, P, C)
    camera = torch.stack([item["camera"] for item in batch])  # (B, 3, H, W)
    semantics = torch.stack([item["semantics"] for item in batch])  # (B, H, W)
    gt_boxes = torch.stack([item["gt_boxes"] for item in batch])  # (B, M, 7)

    # Convert gt_boxes to list of tensors for BEV label generation
    # (because generate_bev_labels_bbox2d expects a list of (N_i, 7))
    gt_boxes_list = [gt_boxes[i] for i in range(gt_boxes.shape[0])]

    # Generate BEV labels once per batch
    heatmap_gt, reg_gt, mask_gt = generate_bev_labels_bbox2d(gt_boxes_list)

    return {
        "points": points,
        "camera": camera,
        "semantics": semantics,
        "gt_boxes": gt_boxes,
        "heatmap_gt": heatmap_gt,
        "reg_gt": reg_gt,
        "mask_gt": mask_gt,
    }


class A2D2Dataset(Dataset):
    """
    PyTorch Dataset for loading A2D2 chunked NPZ files.

    Each NPZ file contains:
        points:    (B, N, C) without timestamps
        camera:    (B, H, W, 3)
        semantics: (B, H, W)
        gt_boxes:  (B, M, 7)

    The dataset flattens all chunks into a global index:
        global_idx → (chunk_idx, frame_idx)
    """

    def __init__(self, root: Path):
        self.root = Path(root)

        # List of NPZ chunk files
        self.chunk_paths: List[Path] = sorted(self.root.glob("*.npz"))
        if not self.chunk_paths:
            raise RuntimeError(f"No .npz files found in {self.root}")

        # Build global index map
        self.index_map: List[Tuple[int, int]] = []
        for ci, path in enumerate(self.chunk_paths):
            with np.load(path) as data:
                num_frames = data["points"].shape[0]
            for fi in range(num_frames):
                self.index_map.append((ci, fi))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a single frame from the appropriate chunk.
        """
        chunk_idx, frame_idx = self.index_map[idx]
        chunk_path = self.chunk_paths[chunk_idx]

        with np.load(chunk_path, allow_pickle=True) as data:
            # LiDAR
            points = torch.from_numpy(data["points"][frame_idx]).half()

            # --- NEW: decode PNG camera bytes ---
            cam_bytes = data["camera"][frame_idx]  # this is a bytes object
            cam_img = Image.open(io.BytesIO(cam_bytes))  # PIL Image
            cam_np = np.array(cam_img)  # (H, W, 3)
            camera = torch.from_numpy(cam_np)
            camera = camera.permute(2, 0, 1).float() / 255.0  # (3, H, W), normalized to [0, 1]
            camera = rescale_image(camera, is_label=False)

            # Semantics (still raw uint8 array)
            semantics = torch.from_numpy(data["semantics"][frame_idx]).clone()
            semantics = rescale_image(semantics, is_label=True)
            semantics[semantics >= params.NUM_SEM_CLASSES] = 255

            # Boxes
            gt_boxes = torch.from_numpy(data["gt_boxes"][frame_idx]).float()

        return {
            "points": points,  # (N, C)
            "camera": camera,  # (3, H, W)
            "semantics": semantics,  # (H, W)
            "gt_boxes": gt_boxes,  # (M, 7)
        }
