import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def inspect_npz(path: str | Path) -> None:
    path = Path(path)
    print(f"\n=== Inspecting NPZ: {path} ===")

    with np.load(path, allow_pickle=True) as data:
        print("\n--- Keys in file ---")
        for k in data.files:
            arr = data[k]
            print(f"{k:12s} shape={arr.shape} dtype={arr.dtype}")

        print("\n--- First frame summary ---")

        # Points
        pts = data["points"][0]
        print(f"points[0] shape: {pts.shape}")
        num_non_zero_pts = np.sum(np.any(pts != 0, axis=1))
        print(f"points[0] non-zero points: {num_non_zero_pts}")

        # Camera
        cam_bytes = data["camera"][0]
        cam = np.array(Image.open(io.BytesIO(cam_bytes)))
        print(f"camera[0] shape: {cam.shape}")

        # Semantics
        sem = data["semantics"][0]
        print(f"semantics[0] shape: {sem.shape}")

        # GT boxes
        boxes = data["gt_boxes"][0]
        print(f"gt_boxes[0] shape: {boxes.shape}")
        print("gt_boxes[0] (non-zero rows):")
        print(boxes[~np.all(boxes == 0, axis=1)])


if __name__ == "__main__":
    inspect_npz(sys.argv[1])
