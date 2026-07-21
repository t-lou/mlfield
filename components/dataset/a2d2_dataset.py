import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from components.definitions.mmperc import MmpercParams
from components.mmperc.label.bev_labels import generate_bev_labels_bbox2d
from components.utils.image_utils import rescale_image


def bev_collate(batch):
    """
    Collate function for A2D2Dataset with fixed-size padded A2D2 tar data.
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
    PyTorch Dataset for reading A2D2 directly from a tar archive.

    This is safe to use with DataLoader workers because each worker opens its
    own tar handle lazily. The main process only indexes the archive and then
    closes the temporary reader.
    """

    def __init__(self, path_tar: Path, params: MmpercParams = MmpercParams()):
        self.path_tar = path_tar
        self.params = params
        self._name = "cam_front_center"
        self._root_parsing = "camera"
        self._tar = None
        self._members: set[str] = set()
        self.color_to_class: Dict[Tuple[int, int, int], int] = {}
        self.class_to_color: Dict[int, Tuple[int, int, int]] = {}
        self.class_to_name: Dict[int, str] = {}

        if not self.path_tar.exists():
            raise RuntimeError(f"A2D2 tar archive not found: {self.path_tar}")

        self._init_archive()
        self.color_key_to_class = {(r << 16) + (g << 8) + b: cid for (r, g, b), cid in self.color_to_class.items()}

    def _init_archive(self) -> None:
        with tarfile.open(self.path_tar, mode="r") as tar:
            self._members = [member.name for member in tar.getmembers()]
            self.color_to_class, self.class_to_color, self.class_to_name = self._load_semantic_mapping(tar)

        # Statistics about the datatypes
        for ext in [".json", ".png", ".npz"]:
            count = sum(1 for p in self._members if p.endswith(ext))
            print(f"Found {count} files with extension {ext} in {self.path_tar}")

        # Filter the members to files only, to match the name, and cluster to measurement time
        assert max(p.count("/") for p in self._members) == 4  # "path/start_time/sensor_type/sensor_name/filename"
        self._members = [p for p in self._members if p.count("/") == 4 and f"/{self._name}/" in p]
        expected_extensions = {"camera": ".png", "label": ".png", "label3D": ".json", "lidar": "npz"}
        self.clustered_paths = {}
        for p in self._members:
            _, timestamp0, sensor_type, _, filename = p.split("/")
            if timestamp0 not in self.clustered_paths:
                self.clustered_paths[timestamp0] = {}
            frame_id = filename.replace("_", ".").split(".")[-2]
            if frame_id not in self.clustered_paths[timestamp0]:
                self.clustered_paths[timestamp0][frame_id] = {}
            expected_extension = expected_extensions[sensor_type]
            if p.endswith(expected_extension):  # Avoid loading additional data
                self.clustered_paths[timestamp0][frame_id][sensor_type] = p

        # Check for completeness and make a list of indices
        # Skip train/val/test splitting as the target is not yet clear
        self.indexing = []
        for timestamp0 in self.clustered_paths:
            for frame_id in self.clustered_paths[timestamp0]:
                assert all(k in self.clustered_paths[timestamp0][frame_id] for k in expected_extensions.keys()), (
                    f"not all keys available in {timestamp0}/{frame_id}: {self.clustered_paths[timestamp0][frame_id]}"
                )
                self.indexing.append((timestamp0, frame_id))
        # Shuffle the indexing randomly to avoid bias from sequential data, with a fixed seed for reproducibility
        random = np.random.RandomState(seed=42)
        random.shuffle(self.indexing)

    def _get_tar(self) -> tarfile.TarFile:
        if self._tar is None:
            self._tar = tarfile.open(self.path_tar, mode="r")
        return self._tar

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_tar"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._tar = None

    def __len__(self) -> int:
        return len(self.indexing)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        timestamp0, frame_id = self.indexing[idx]
        frame = self._load_frame(self.clustered_paths[timestamp0][frame_id])

        points = torch.from_numpy(frame["points"]).half()
        camera = frame["camera"]
        semantics = torch.from_numpy(frame["semantics"]).clone()
        semantics = rescale_image(semantics, scale_factor=self.params.image_scale, is_label=True)
        semantics[semantics >= self.params.num_sem_classes] = 255
        gt_boxes = torch.from_numpy(frame["gt_boxes"]).float()

        return {
            "points": points,
            "camera": camera,
            "semantics": semantics,
            "semantics_mapping_color": list(self.class_to_color.items()),
            "semantics_mapping_name": list(self.class_to_name.items()),
            "gt_boxes": gt_boxes,
        }

    def __del__(self) -> None:
        try:
            if self._tar is not None:
                self._tar.close()
        except Exception:
            pass

    def find_fc_files(self) -> List[str]:
        """
        Find all front-center camera PNGs inside the tar.
        """
        fc_frames: List[str] = []

        for name in self._members:
            path = PurePosixPath(name)
            if len(path.parts) >= 4:
                sensor_type, cam_name = path.parts[2], path.parts[3]
                if sensor_type == self._root_parsing and cam_name == self._name and path.suffix.lower() == ".png":
                    fc_frames.append(name)

        return fc_frames

    def load_img(self, path: PurePosixPath, mode: str) -> np.ndarray:
        tar = self._get_tar()
        fileobj = tar.extractfile(str(path))
        if fileobj is None:
            raise FileNotFoundError(f"Image file {path} not found in tar")

        with fileobj:
            img = Image.open(io.BytesIO(fileobj.read())).convert(mode)
        return np.array(img)

    def load_lidar(self, path: PurePosixPath) -> Tuple[np.ndarray, np.ndarray]:
        tar = self._get_tar()
        fileobj = tar.extractfile(str(path))
        if fileobj is None:
            raise FileNotFoundError(f"LIDAR file {path} not found in tar")

        with fileobj:
            data = np.load(io.BytesIO(fileobj.read()))

        points = data["points"]
        reflectance = data["reflectance"][:, None]
        arr = np.concatenate([points, reflectance], axis=1).astype(np.float16)

        padded = np.zeros((self.params.num_lidar_points, arr.shape[1]), dtype=arr.dtype)
        n = min(arr.shape[0], self.params.num_lidar_points)
        padded[:n] = arr[:n]

        padded_timestamp = np.zeros((self.params.num_lidar_points,), dtype=np.int64)
        padded_timestamp[:n] = data["timestamp"][:n]
        return padded, padded_timestamp

    def load_boxes(self, path: PurePosixPath) -> np.ndarray:
        tar = self._get_tar()
        fileobj = tar.extractfile(str(path))
        if fileobj is None:
            raise FileNotFoundError(f"3D box file {path} not found in tar")

        with fileobj:
            data = json.load(io.BytesIO(fileobj.read()))

        boxes: List[List[float]] = []
        for obj in data.values():
            center = obj["center"]
            size = obj["size"]
            yaw = obj.get("rot_angle", 0.0)
            boxes.append(center + size + [yaw])

        padded = np.zeros((self.params.num_gt_boxes, 7), dtype=np.float32)
        n = min(len(boxes), self.params.num_gt_boxes)
        if n > 0:
            padded[:n] = np.array(boxes[:n], dtype=np.float32)

        return padded

    def _load_semantic_mapping(
        self, tar: tarfile.TarFile
    ) -> Tuple[Dict[Tuple[int, int, int], int], Dict[int, Tuple[int, int, int]], Dict[int, str]]:
        json_candidates = [name for name in self._members if name.endswith("class_list.json")]
        if len(json_candidates) != 1:
            raise FileNotFoundError("A2D2 class_list.json not found in tar or multiple candidates found.")

        json_path = json_candidates[0]
        fileobj = tar.extractfile(json_path)
        if fileobj is None:
            raise FileNotFoundError(f"Could not extract {json_path}")

        with fileobj:
            data = json.load(io.BytesIO(fileobj.read()))

        color_to_class: Dict[Tuple[int, int, int], int] = {}
        class_to_color: Dict[int, Tuple[int, int, int]] = {}
        class_to_name: Dict[int, str] = {}

        for cid, (hex_color, name) in enumerate(data.items()):
            hex_color = hex_color.lstrip("#")
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            color_to_class[rgb] = cid
            class_to_color[cid] = rgb
            class_to_name[cid] = name

        return color_to_class, class_to_color, class_to_name

    def convert_semantic_rgb_to_class(self, rgb_img: np.ndarray) -> np.ndarray:
        H, W, _ = rgb_img.shape
        class_mask = np.zeros((H, W), dtype=np.uint8)
        flat = rgb_img.reshape(-1, 3).astype(np.uint32)
        out = class_mask.reshape(-1)
        keys = (flat[:, 0] << 16) + (flat[:, 1] << 8) + flat[:, 2]

        for key, cid in self.color_key_to_class.items():
            out[keys == key] = cid

        return class_mask

    def _build_paths(self, fc_path: str) -> Dict[str, PurePosixPath]:
        p = PurePosixPath(fc_path)
        parts = list(p.parts)
        if len(parts) < 4:
            raise ValueError(f"Unexpected frame path structure: {fc_path}")
        if parts[2] != self._root_parsing:
            raise ValueError(f"Expected camera path in position 2, got {parts[2]} from {fc_path}")

        def with_kind(kind: str, suffix: str) -> PurePosixPath:
            new = list(parts)
            new[2] = kind
            return PurePosixPath(*new).with_suffix(suffix)

        paths = {
            "camera": p,
            "lidar": with_kind("lidar", ".npz"),
            "semantics": with_kind("label", ".png"),
            "gt_boxes": with_kind("label3D", ".json"),
        }

        for key, path in paths.items():
            if str(path) not in self._members:
                msg = f"{key} file {path} not found in tar, expected paths:\n\t"
                msg += "\n\t".join([f"{k}: {v}" for k, v in paths.items()])
                raise FileNotFoundError(msg)

        return paths

    def _load_frame(self, paths: dict[str, str]) -> Dict[str, np.ndarray]:
        # The paths are indexed with "camera", "label", "label3D", "lidar"
        sem_rgb = self.load_img(paths["label"], mode="RGB")
        sem_class = self.convert_semantic_rgb_to_class(sem_rgb)

        lidar_xyzi, lidar_timestamp = self.load_lidar(paths["lidar"])
        camera_img = self.load_img(paths["camera"], mode="RGB")

        camera = torch.from_numpy(camera_img).permute(2, 0, 1).float() / 255.0
        camera = rescale_image(camera, scale_factor=self.params.image_scale, is_label=False)

        return {
            "points": lidar_xyzi,
            "points_timestamp": lidar_timestamp,
            "camera": camera,
            "semantics": sem_class,
            "gt_boxes": self.load_boxes(paths["label3D"]),
        }
