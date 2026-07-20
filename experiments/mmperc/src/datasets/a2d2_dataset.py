import io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import common.params as params
from common.utils import rescale_image
from label.bev_labels import generate_bev_labels_bbox2d


def bev_collate(batch):
    """
    Collate function for A2D2Dataset with fixed-size padded NPZ data.
    """

    points = torch.stack([item["points"] for item in batch])  # (B, P, C)
    camera = torch.stack([item["camera"] for item in batch])  # (B, 3, H, W)
    semantics = torch.stack([item["semantics"] for item in batch])  # (B, H, W)
    gt_boxes = torch.stack([item["gt_boxes"] for item in batch])  # (B, M, 7)

    semantics_mapping_color = [item["semantics_mapping_color"] for item in batch]
    semantics_mapping_name = [item["semantics_mapping_name"] for item in batch]

    gt_boxes_list = [gt_boxes[i] for i in range(gt_boxes.shape[0])]
    heatmap_gt, reg_gt, mask_gt = generate_bev_labels_bbox2d(gt_boxes_list)

    return {
        "points": points,
        "camera": camera,
        "semantics": semantics,
        "semantics_mapping_color": semantics_mapping_color,
        "semantics_mapping_name": semantics_mapping_name,
        "gt_boxes": gt_boxes,
        "heatmap_gt": heatmap_gt,
        "reg_gt": reg_gt,
        "mask_gt": mask_gt,
    }


class A2D2Dataset(Dataset):
    """
    PyTorch Dataset for loading A2D2 chunked NPZ files.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

        self.chunk_paths: List[Path] = sorted(self.root.glob("*.npz"))
        if not self.chunk_paths:
            raise RuntimeError(f"No .npz files found in {self.root}")

        self.index_map: List[Tuple[int, int]] = []
        for ci, path in enumerate(self.chunk_paths):
            with np.load(path, allow_pickle=True) as data:
                num_frames = data["points"].shape[0]
            for fi in range(num_frames):
                self.index_map.append((ci, fi))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chunk_idx, frame_idx = self.index_map[idx]
        chunk_path = self.chunk_paths[chunk_idx]

        with np.load(chunk_path, allow_pickle=True) as data:
            # LiDAR
            points = torch.from_numpy(data["points"][frame_idx]).half()

            # Camera (PNG bytes → RGB tensor)
            cam_bytes = data["camera"][frame_idx]
            with Image.open(io.BytesIO(cam_bytes)) as cam_img:
                cam_img = cam_img.convert("RGB")
                cam_np = np.asarray(cam_img, dtype=np.uint8).copy()

            camera = torch.from_numpy(cam_np).permute(2, 0, 1).float() / 255.0
            camera = rescale_image(camera, is_label=False)

            # Semantics
            semantics = torch.from_numpy(data["semantics"][frame_idx]).clone()
            semantics = rescale_image(semantics, is_label=True)
            semantics[semantics >= params.NUM_SEM_CLASSES] = 255

            # ---------------------------------------------------------
            # Semantic mappings (global, not per-frame)
            # ---------------------------------------------------------
            semantics_mapping_color = list(data["semantics_mapping_color"])
            semantics_mapping_name = list(data["semantics_mapping_name"])

            # Boxes
            gt_boxes = torch.from_numpy(data["gt_boxes"][frame_idx]).float()

        return {
            "points": points,
            "camera": camera,
            "semantics": semantics,
            "semantics_mapping_color": semantics_mapping_color,
            "semantics_mapping_name": semantics_mapping_name,
            "gt_boxes": gt_boxes,
        }
