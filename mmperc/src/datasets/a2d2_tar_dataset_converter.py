import io
import json
import logging
import os
import random
import shutil
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from common.utils import encode_png_array


class A2D2TarDatasetConverter:
    """
    Convert A2D2 tar archives into compact, batched NPZ files.

    Each batch contains:
        - points:    (B, num_lidar_points, C)
        - camera:    (B, H, W, 3)
        - semantics: (B, H, W)
        - gt_boxes:  (B, num_gt_boxes, 7)
    """

    def __init__(
        self,
        tar_path: str,
        group_size: int = 200,
        num_lidar_points: int = 12_000,
        num_gt_boxes: int = 200,
    ) -> None:
        self.tar_path = tar_path
        self.tar = tarfile.open(tar_path, "r")

        # A2D2 dataset structure parameters
        self._name = "cam_front_center"
        self._root_parsing = "camera"
        self._dataset_type = "camera_lidar_semantic_bboxes"

        # batching parameters
        self._group_size = group_size
        self._num_lidar_points = num_lidar_points
        self._num_gt_boxes = num_gt_boxes

        # Cache all member names for fast lookup (important on WSL)
        self._members = {m.name for m in self.tar.getmembers()}

        # Load semantic mapping from color to class ID
        self.color_to_class, self.class_to_color, self.class_to_name = self._load_semantic_mapping()

        logging.info(f"Tar file {self.tar_path} parsed.")

    def close(self) -> None:
        """Close the underlying tar file."""
        self.tar.close()

    # ----------------------------------------------------------------------
    # Discovery
    # ----------------------------------------------------------------------

    def find_fc_files(self) -> List[str]:
        """
        Find all front-center camera PNGs inside the tar.
        """
        fc_frames: List[str] = []

        for name in self._members:
            p = PurePosixPath(name)
            if len(p.parts) >= 4:
                sensor_type, cam_name = p.parts[2], p.parts[3]
                if sensor_type == self._root_parsing and cam_name == self._name and p.suffix.lower() == ".png":
                    fc_frames.append(name)

        return fc_frames

    def shuffle_and_group_pngs(self) -> List[List[str]]:
        """
        Shuffle and group front-center PNGs into fixed-size batches.
        """
        fc_frames = self.find_fc_files()
        random.shuffle(fc_frames)

        groups = [fc_frames[i : i + self._group_size] for i in range(0, len(fc_frames), self._group_size)]

        # Drop incomplete final group
        if groups and len(groups[-1]) < self._group_size:
            groups = groups[:-1]

        return groups

    # ----------------------------------------------------------------------
    # Loading helpers
    # ----------------------------------------------------------------------

    def load_img(self, path: PurePosixPath, mode: str) -> np.ndarray:
        """
        Load an image from the tar and return as a NumPy array.
        """
        fileobj = self.tar.extractfile(str(path))
        if fileobj is None:
            raise FileNotFoundError(f"Image file {path} not found in tar")

        img = Image.open(io.BytesIO(fileobj.read())).convert(mode)
        return np.array(img)

    def load_lidar(self, path: PurePosixPath) -> np.ndarray:
        """
        Load lidar .npz from tar, pad/truncate to fixed number of points.
        """
        fileobj = self.tar.extractfile(str(path))
        if fileobj is None:
            raise FileNotFoundError(f"LIDAR file {path} not found in tar")

        data = np.load(io.BytesIO(fileobj.read()))
        points = data["points"]
        reflectance = data["reflectance"][:, None]

        arr = np.concatenate([points, reflectance], axis=1)
        arr = arr.astype(np.float16)
        logging.debug(f"Loaded LIDAR from {path} with shape {arr.shape}")

        padded = np.zeros((self._num_lidar_points, arr.shape[1]), dtype=arr.dtype)
        n = min(arr.shape[0], self._num_lidar_points)
        padded[:n] = arr[:n]

        # timestamp are huge int64 values, ignore for now
        padded_timestamp = np.zeros((self._num_lidar_points,), dtype=np.int64)
        padded_timestamp[:n] = data["timestamp"][:n]
        return padded, padded_timestamp

    def load_boxes(self, path: PurePosixPath) -> np.ndarray:
        """
        Load 3D bounding boxes from JSON and pad to fixed count.
        """
        fileobj = self.tar.extractfile(str(path))
        if fileobj is None:
            raise FileNotFoundError(f"3D box file {path} not found in tar")

        data = json.load(io.BytesIO(fileobj.read()))
        boxes: List[List[float]] = []

        for obj in data.values():
            center = obj["center"]  # [x, y, z]
            size = obj["size"]  # [dx, dy, dz]
            yaw = obj.get("rot_angle", 0.0)
            boxes.append(center + size + [yaw])

        padded = np.zeros((self._num_gt_boxes, 7), dtype=np.float32)
        n = min(len(boxes), self._num_gt_boxes)
        if n > 0:
            padded[:n] = np.array(boxes[:n], dtype=np.float32)

        return padded

    def _load_semantic_mapping(self) -> Dict[Tuple[int, int, int], int]:
        """
        Load A2D2 class_list.json from the tar archive and convert hex colors to RGB → class_id mapping.
        """
        # Try to find the JSON inside the tar
        json_candidates = [name for name in self._members if name.endswith("class_list.json")]

        if len(json_candidates) != 1:
            raise FileNotFoundError("A2D2 class_list.json not found in tar or multiple candidates found.")

        json_path = json_candidates[0]
        fileobj = self.tar.extractfile(json_path)
        if fileobj is None:
            raise FileNotFoundError(f"Could not extract {json_path}")

        data = json.load(io.BytesIO(fileobj.read()))

        color_to_class = {}
        class_to_name = {}
        class_to_color = {}

        # Assign class IDs in the order they appear
        for cid, (hex_color, name) in enumerate(data.items()):
            # Convert "#rrggbb" → (R,G,B)
            hex_color = hex_color.lstrip("#")
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

            color_to_class[rgb] = cid
            class_to_color[cid] = rgb
            class_to_name[cid] = name

        return color_to_class, class_to_color, class_to_name

    def convert_semantic_rgb_to_class(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Convert an RGB semantic mask (H, W, 3) into class IDs (H, W).
        """
        H, W, _ = rgb_img.shape
        class_mask = np.zeros((H, W), dtype=np.uint8)

        # Flatten for vectorized matching
        flat = rgb_img.reshape(-1, 3).astype(np.uint32)
        out = class_mask.reshape(-1)

        # Build a lookup table for speed
        # Convert RGB triplet → int key
        keys = (flat[:, 0] << 16) + (flat[:, 1] << 8) + flat[:, 2]

        # Precompute mapping in same integer space
        lut = {}
        for (r, g, b), cid in self.color_to_class.items():
            lut[(r << 16) + (g << 8) + b] = cid

        # Vectorized mapping
        for k, cid in lut.items():
            out[keys == k] = cid

        return class_mask

    # ----------------------------------------------------------------------
    # Path resolution
    # ----------------------------------------------------------------------

    def _build_paths(self, fc_path: str) -> Dict[str, PurePosixPath]:
        """
        Given a camera PNG path, build corresponding lidar, semantic, and box paths.
        """
        p = PurePosixPath(fc_path)
        parts = p.parts

        def conv(kind: str) -> PurePosixPath:
            new = list(parts)
            new[1:] = [part.replace("camera", kind) for part in new[1:]]
            return PurePosixPath(*new)

        paths = {
            "camera": p,
            "lidar": conv("lidar").with_suffix(".npz"),
            "semantics": conv("label").with_suffix(".png"),
            "gt_boxes": conv("label3D").with_suffix(".json"),
        }

        # Validate existence
        for key, path in paths.items():
            if str(path) not in self._members:
                raise FileNotFoundError(f"{key} file {path} not found in tar")

        return paths

    # ----------------------------------------------------------------------
    # Frame and batch processing
    # ----------------------------------------------------------------------

    def proceed_one_frame(self, fc_path: str) -> Dict[str, np.ndarray]:
        """
        Load all modalities for a single frame.
        """
        paths = self._build_paths(fc_path)

        # Load semantic PNG as RGB
        sem_rgb = self.load_img(paths["semantics"], mode="RGB")
        # Convert to class IDs
        sem_class = self.convert_semantic_rgb_to_class(sem_rgb)

        # Read lidar points and timestamp separately
        lidar_xyzi, lidar_timestamp = self.load_lidar(paths["lidar"])

        return {
            "points": lidar_xyzi,
            "points_timestamp": lidar_timestamp,
            "camera": self.load_img(paths["camera"], mode="RGB"),
            "semantics": sem_class,
            "gt_boxes": self.load_boxes(paths["gt_boxes"]),
        }

    def proceed_one_bunch(self, fc_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Load and stack a batch of frames into compact arrays.
        """
        assert len(fc_paths) == self._group_size, "Incorrect group size"

        frames = [self.proceed_one_frame(p) for p in fc_paths]

        # Extract raw camera arrays
        camera_arrays = [f["camera"] for f in frames]

        # Parallel PNG encoding
        max_workers = min(os.cpu_count(), 16)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            camera_pngs = list(executor.map(encode_png_array, camera_arrays))

        # Raw semantic labels
        sem_raw = np.stack([f["semantics"] for f in frames], axis=0)

        # Convert dicts → list of pairs (portable, no pickle)
        class_to_color_list = [(cid, rgb) for cid, rgb in self.class_to_color.items()]
        class_to_name_list = [(cid, name) for cid, name in self.class_to_name.items()]

        return {
            "points": np.stack([f["points"] for f in frames], axis=0),
            "points_timestamp": np.stack([f["points_timestamp"] for f in frames], axis=0),
            "camera": np.array(camera_pngs, dtype=object),
            "semantics": sem_raw,
            "semantics_mapping_color": np.array(class_to_color_list, dtype=object),
            "semantics_mapping_name": np.array(class_to_name_list, dtype=object),
            "gt_boxes": np.stack([f["gt_boxes"] for f in frames], axis=0),
        }

    # ----------------------------------------------------------------------
    # Saving
    # ----------------------------------------------------------------------

    def save_bunch(self, batch: Dict[str, np.ndarray], output_dir: Path, idx: int) -> Path:
        """
        Save a batch to a compressed NPZ file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"a2d2_{idx:04d}.npz"

        np.savez_compressed(
            out_path,
            points=batch["points"],
            points_timestamp=batch["points_timestamp"],
            camera=batch["camera"],
            semantics=batch["semantics"],
            gt_boxes=batch["gt_boxes"],
        )

        logging.info(f"Saved {out_path} with batch size {batch['points'].shape[0]}")
        return out_path


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------

if __name__ == "__main__":
    tar_path = sys.argv[1]
    output_path = Path(sys.argv[2])

    # Clean output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    converter = A2D2TarDatasetConverter(tar_path)

    try:
        groups = converter.shuffle_and_group_pngs()

        # tqdm shows progress, ETA, speed, etc.
        for idx, group in enumerate(tqdm(groups, desc="Converting groups"), start=1):
            batch = converter.proceed_one_bunch(group)
            converter.save_bunch(batch, output_path, idx)

    finally:
        converter.close()
