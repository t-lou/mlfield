import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class A2D2Dataset(Dataset):
    """
    A2D2 dataset loader for directory structures like:

        camera_lidar_semantic_bboxes/
          20180807_145028/
            camera/cam_front_center/*.png
            lidar/cam_front_center/*.npz
            label/cam_front_center/*.png
            label3D/cam_front_center/*.json

    Loads:
        - Lidar point clouds (npz)
        - Camera images or precomputed camera tokens
        - Semantic segmentation masks
        - 3D bounding boxes
    """

    def __init__(
        self,
        root: str | Path,
        use_cam_tokens: bool = False,
        transform=None,
        sub_name: str = "cam_front_center",
        num_lidar_points: int = 10_000,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.use_cam_tokens = use_cam_tokens
        self.transform = transform

        self.sub_name = sub_name
        self.num_lidar_points = num_lidar_points
        self.to_tensor = transforms.ToTensor()

        # ------------------------------------------------------------
        # Build sample index
        # ------------------------------------------------------------
        self.samples: list[tuple[str, str]] = []

        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue

            lidar_dir = folder / "lidar" / self.sub_name
            for file in sorted(lidar_dir.glob("*.npz")):
                # Example filename:
                #   20180807_145028_lidar_frontcenter_000000123.npz
                # → base = 000000123
                base = file.name.replace("_lidar_frontcenter_", "_")
                self.samples.append((folder.name, base))

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    @staticmethod
    def _find_only_file_with_ext(directory: Path, ext: str) -> Path:
        files = list(directory.glob(f"*.{ext}"))
        if len(files) != 1:
            raise ValueError(f"Expected exactly one .{ext} file in {directory}, found {len(files)}")
        return files[0]

    # ------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------
    def load_lidar(self, folder: str, base: str) -> torch.Tensor:
        path = self._find_only_file_with_ext(self.root / folder / "lidar" / self.sub_name, "npz")

        data = np.load(path)
        points = data["points"]
        reflectance = data["reflectance"][:, None]
        timestamp = data["timestamp"][:, None]

        arr = np.concatenate([points, reflectance, timestamp], axis=1)
        logging.debug(f"Loaded LIDAR from {path} with shape {arr.shape}")

        # Pad or truncate to fixed size
        num = min(arr.shape[0], self.num_lidar_points)
        padded = np.zeros((self.num_lidar_points, arr.shape[1]), dtype=arr.dtype)
        padded[:num] = arr[:num]

        return torch.tensor(padded, dtype=torch.float32)

    def load_camera(self, folder: str, base: str) -> torch.Tensor:
        if self.use_cam_tokens:
            path = self._find_only_file_with_ext(self.root / folder / "camera_tokens" / self.sub_name, "npz")
            arr = np.load(path)
            return torch.tensor(arr, dtype=torch.float32)

        path = self._find_only_file_with_ext(self.root / folder / "camera" / self.sub_name, "png")
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img)

    def load_semantics(self, folder: str, base: str) -> torch.Tensor:
        path = self._find_only_file_with_ext(self.root / folder / "label" / self.sub_name, "png")
        arr = np.array(Image.open(path))
        return torch.tensor(arr, dtype=torch.long).unsqueeze(0)

    def load_boxes(self, folder: str, base: str) -> torch.Tensor:
        path = self._find_only_file_with_ext(self.root / folder / "label3D" / self.sub_name, "json")

        with open(path, "r") as f:
            data = json.load(f)

        boxes = []
        for obj in data.get("objects", []):
            center = obj["center"]
            size = obj["size"]
            yaw = obj["rotation"]
            boxes.append(center + size + [yaw])

        if not boxes:
            return torch.zeros((0, 7), dtype=torch.float32)

        return torch.tensor(boxes, dtype=torch.float32)

    # ------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        folder, base = self.samples[idx]

        points = self.load_lidar(folder, base)
        cam = self.load_camera(folder, base)
        semantics = self.load_semantics(folder, base)
        boxes = self.load_boxes(folder, base)

        sample = {
            "points": points,
            "cam_tokens": cam,
            "semantics": semantics,
            "gt_boxes": boxes,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
