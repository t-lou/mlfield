# src/../data
# |-- a2d2-preview
# |   |-- LICENSE.txt
# |   |-- README.txt
# |   |-- camera_lidar
# |   |-- camera_lidar_semantic
# |   |-- camera_lidar_semantic_bboxes
# |   |-- cams_lidars.json
# |   `-- tutorial.ipynb
# `-- a2d2-preview.tar

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
    A2D2 loader for folders like:
    camera_lidar_semantic_bboxes/
      20180807_145028/
        camera/cam_front_center/*.png
        lidar/cam_front_center/*.npz
        label/cam_front_center/*.png
        label3D/cam_front_center/*.json
    """

    def __init__(self, root, use_cam_tokens=False, transform=None):
        self.root = Path(root)
        self.use_cam_tokens = use_cam_tokens
        self.transform = transform

        self._sub_name = "cam_front_center"
        self._num_lidar_points = 10_000
        self._img_to_tensor = transforms.ToTensor()

        self.samples = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            lidar_dir = folder / "lidar" / self._sub_name
            for file in sorted(lidar_dir.glob("*.npz")):
                base = file.name.replace("_lidar_frontcenter_", "_")
                self.samples.append((folder.name, base))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _find_only_file_with_ext(directory, ext):
        files = list(directory.glob(f"*.{ext}"))
        if len(files) != 1:
            raise ValueError(f"Expected exactly one .{ext} file in {directory}, found {len(files)}")
        return files[0]

    def load_lidar(self, folder, base):
        path = self._find_only_file_with_ext(self.root / folder / "lidar" / self._sub_name, "npz")
        arr = np.load(path)["points"]
        logging.error(f"Loaded LIDAR from {path} with shape {arr.shape}")
        num_points = min(arr.shape[0], self._num_lidar_points)
        pad = np.zeros((self._num_lidar_points, arr.shape[1]), dtype=arr.dtype)
        pad[:num_points] = arr[:num_points]

        return torch.tensor(pad, dtype=torch.float32)

    def load_camera(self, folder, base):
        if self.use_cam_tokens:
            path = self._find_only_file_with_ext(self.root / folder / "camera_tokens" / self._sub_name, "npz")
            arr = np.load(path)
            return torch.tensor(arr, dtype=torch.float32)
        else:
            path = self._find_only_file_with_ext(self.root / folder / "camera" / self._sub_name, "png")
            img = Image.open(path).convert("RGB")
            return self._img_to_tensor(img)  # now it's a tensor (3, H, W)

    def load_semantics(self, folder, base):
        path = self._find_only_file_with_ext(self.root / folder / "label" / self._sub_name, "png")
        arr = np.array(Image.open(path))
        return torch.tensor(arr, dtype=torch.long).unsqueeze(0)

    def load_boxes(self, folder, base):
        path = self._find_only_file_with_ext(self.root / folder / "label3D" / self._sub_name, "json")
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

    def __getitem__(self, idx):
        folder, base = self.samples[idx]

        points = self.load_lidar(folder, base)
        cam = self.load_camera(folder, base)
        semantics = self.load_semantics(folder, base)
        boxes = self.load_boxes(folder, base)

        sample = {"points": points, "cam_tokens": cam, "semantics": semantics, "gt_boxes": boxes}

        if self.transform:
            sample = self.transform(sample)

        return sample
