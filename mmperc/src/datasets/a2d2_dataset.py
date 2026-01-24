import io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import common.params as params
from common.utils import rescale_image

# local dependencies
from label.bev_labels import generate_bev_labels_bbox2d


def bev_collate(batch):
    """
    Collate function for A2D2Dataset with fixed-size padded NPZ data.

    Input batch: list of dicts, each containing:
        points:                    (num_lidar_points, C)
        camera:                    (3, H, W)
        semantics:                 (H, W)
        semantics_mapping_color:   list[(cid, (R,G,B))]
        semantics_mapping_name:    list[(cid, name)]
        gt_boxes:                  (num_gt_boxes, 7)

    Output:
        points:                    (B, num_lidar_points, C)
        camera:                    (B, 3, H, W)
        semantics:                 (B, H, W)
        semantics_mapping_color:   list of lists
        semantics_mapping_name:    list of lists
        gt_boxes:                  (B, num_gt_boxes, 7)
        heatmap_gt:                (B, 1, H_bev, W_bev)
        reg_gt:                    (B, 6, H_bev, W_bev)
        mask_gt:                   (B, 1, H_bev, W_bev)
    """

    # ---------------------------------------------------------
    # Fixed-size tensors → stack directly
    # ---------------------------------------------------------
    points = torch.stack([item["points"] for item in batch])  # (B, P, C)
    camera = torch.stack([item["camera"] for item in batch])  # (B, 3, H, W)
    semantics = torch.stack([item["semantics"] for item in batch])  # (B, H, W)
    gt_boxes = torch.stack([item["gt_boxes"] for item in batch])  # (B, M, 7)

    # ---------------------------------------------------------
    # Semantic mappings (variable-size lists → keep as list)
    # ---------------------------------------------------------
    semantics_mapping_color = [item["semantics_mapping_color"] for item in batch]
    semantics_mapping_name = [item["semantics_mapping_name"] for item in batch]

    # ---------------------------------------------------------
    # Convert gt_boxes to list of tensors for BEV label generation
    # ---------------------------------------------------------
    gt_boxes_list = [gt_boxes[i] for i in range(gt_boxes.shape[0])]

    # ---------------------------------------------------------
    # Generate BEV labels once per batch
    # ---------------------------------------------------------
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
            # ---------------------------------------------------------
            # LiDAR
            # ---------------------------------------------------------
            points = torch.from_numpy(data["points"][frame_idx]).half()

            # ---------------------------------------------------------
            # Camera (PNG bytes → RGB tensor)
            # ---------------------------------------------------------
            cam_bytes = data["camera"][frame_idx]  # bytes
            cam_img = Image.open(io.BytesIO(cam_bytes))  # PIL Image
            cam_np = np.array(cam_img)  # (H, W, 3)

            camera = torch.from_numpy(cam_np).permute(2, 0, 1).float() / 255.0
            camera = rescale_image(camera, is_label=False)

            # ---------------------------------------------------------
            # Semantics (class IDs)
            # ---------------------------------------------------------
            semantics = torch.from_numpy(data["semantics"][frame_idx]).clone()
            semantics = rescale_image(semantics, is_label=True)

            # Clamp invalid labels
            semantics[semantics >= params.NUM_SEM_CLASSES] = 255

            # ---------------------------------------------------------
            # Semantic mappings (list of pairs)
            # ---------------------------------------------------------
            # Each is stored as dtype=object array of (cid, value)
            mapping_color_np = data["semantics_mapping_color"]
            mapping_name_np = data["semantics_mapping_name"]

            # Convert numpy object arrays → Python lists
            semantics_mapping_color = list(mapping_color_np)
            semantics_mapping_name = list(mapping_name_np)

            # ---------------------------------------------------------
            # Boxes
            # ---------------------------------------------------------
            gt_boxes = torch.from_numpy(data["gt_boxes"][frame_idx]).float()

        return {
            "points": points,  # (N, C)
            "camera": camera,  # (3, H, W)
            "semantics": semantics,  # (H, W)
            "semantics_mapping_color": semantics_mapping_color,  # list[(cid, (R,G,B))]
            "semantics_mapping_name": semantics_mapping_name,  # list[(cid, name)]
            "gt_boxes": gt_boxes,  # (M, 7)
        }
