from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class A2D2Dataset(Dataset):
    """
    PyTorch Dataset for loading A2D2 chunked NPZ files.

    Each NPZ file contains:
        points:    (B, N, C)
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

        with np.load(chunk_path) as data:
            # Convert to tensors
            points = torch.from_numpy(data["points"][frame_idx]).float()
            camera = torch.from_numpy(data["camera"][frame_idx]).permute(2, 0, 1).float()
            semantics = torch.from_numpy(data["semantics"][frame_idx]).long()
            gt_boxes = torch.from_numpy(data["gt_boxes"][frame_idx]).float()

        return {
            "points": points,  # (N, C)
            "camera": camera,  # (3, H, W)
            "semantics": semantics,  # (H, W)
            "gt_boxes": gt_boxes,  # (M, 7)
        }
