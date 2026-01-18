import io
import json
import logging
import random
import sys
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


class A2D2TarDatasetConverter:
    def __init__(
        self,
        tar_path: str,
        group_size: int = 200,
        num_lidar_points: int = 12_000,
        num_gt_boxes: int = 200,
    ):
        self.tar_path = tar_path
        self.tar = tarfile.open(tar_path, "r")

        self._name = "cam_front_center"
        self._root_parsing = "camera"
        self._dataset_type = "camera_lidar_semantic_bboxes"
        self._group_size = group_size
        self._num_lidar_points = num_lidar_points
        self._num_gt_boxes = num_gt_boxes

        # cache member names for faster lookup on WSL
        self._members = {m.name for m in self.tar.getmembers()}

    def close(self):
        self.tar.close()

    def find_fc_files(self) -> list[str]:
        fc_frames = []
        for name in self._members:
            p = PurePosixPath(name)
            if len(p.parts) >= 4:
                sensor_type, cam_name = p.parts[2], p.parts[3]
                if (
                    sensor_type == self._root_parsing
                    and cam_name == self._name
                    and p.parts[-1].lower().endswith(".png")
                ):
                    fc_frames.append(name)
        return fc_frames

    def load_img(self, path: PurePosixPath, mode: str) -> np.ndarray:
        fileobj = self.tar.extractfile(str(path))
        if fileobj is None:
            raise FileNotFoundError(f"Image file {path} not found in tar")
        img = Image.open(io.BytesIO(fileobj.read())).convert(mode)
        return np.array(img)

    def load_lidar(self, path: PurePosixPath) -> np.ndarray:
        fileobj = self.tar.extractfile(str(path))
        if fileobj is None:
            raise FileNotFoundError(f"LIDAR file {path} not found in tar")

        data = np.load(io.BytesIO(fileobj.read()))
        points = data["points"]
        reflectance = data["reflectance"][:, None]
        timestamp = data["timestamp"][:, None]

        arr = np.concatenate([points, reflectance, timestamp], axis=1)
        logging.debug(f"Loaded LIDAR from {path} with shape {arr.shape}")

        padded = np.zeros((self._num_lidar_points, arr.shape[1]), dtype=arr.dtype)
        n = min(arr.shape[0], self._num_lidar_points)
        padded[:n] = arr[:n]
        return padded

    def load_boxes(self, path: PurePosixPath) -> np.ndarray:
        fileobj = self.tar.extractfile(str(path))
        if fileobj is None:
            raise FileNotFoundError(f"3D box file {path} not found in tar")

        data = json.load(io.BytesIO(fileobj.read()))
        boxes = []

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

    def shuffle_and_group_pngs(self) -> list[list[str]]:
        fc_frames = self.find_fc_files()
        random.shuffle(fc_frames)

        grouped = [fc_frames[i : i + self._group_size] for i in range(0, len(fc_frames), self._group_size)]
        if grouped and len(grouped[-1]) < self._group_size:
            grouped = grouped[:-1]
        return grouped

    def _build_paths(self, fc_path: str) -> dict:
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

        for k, v in paths.items():
            if str(v) not in self._members:
                raise FileNotFoundError(f"{k} file {v} not found in tar")
        return paths

    def proceed_one_frame(self, fc_path: str) -> dict:
        paths = self._build_paths(fc_path)

        points = self.load_lidar(paths["lidar"])
        cam = self.load_img(paths["camera"], mode="RGB")
        semantics = self.load_img(paths["semantics"], mode="L")  # label IDs
        boxes = self.load_boxes(paths["gt_boxes"])

        return {
            "points": points,  # (num_lidar_points, C)
            "camera": cam,  # (H, W, 3)
            "semantics": semantics,  # (H, W)
            "gt_boxes": boxes,  # (num_gt_boxes, 7)
        }

    def proceed_one_bunch(self, fc_paths: list[str]) -> dict:
        assert len(fc_paths) == self._group_size, "Input group size does not match expected group size."

        frames = [self.proceed_one_frame(p) for p in fc_paths]

        # stack for compact, fast loading
        points = np.stack([f["points"] for f in frames], axis=0)
        camera = np.stack([f["camera"] for f in frames], axis=0)
        semantics = np.stack([f["semantics"] for f in frames], axis=0)
        gt_boxes = np.stack([f["gt_boxes"] for f in frames], axis=0)

        return {
            "points": points,  # (B, num_lidar_points, C)
            "camera": camera,  # (B, H, W, 3)
            "semantics": semantics,  # (B, H, W)
            "gt_boxes": gt_boxes,  # (B, num_gt_boxes, 7)
        }

    def save_bunch(self, batch: dict, output_dir: Path, idx: int) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"a2d2_{idx:04d}.npz"

        # compressed for disk + WSL friendliness
        np.savez_compressed(
            out_path,
            points=batch["points"],
            camera=batch["camera"],
            semantics=batch["semantics"],
            gt_boxes=batch["gt_boxes"],
        )

        logging.info(f"Saved {out_path} with batch size {batch['points'].shape[0]}")
        return out_path


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    tar_path = sys.argv[1]
    output_path = Path(sys.argv[2])

    # 1. If the folder exists, delete it recursively
    if output_path.exists():
        shutil.rmtree(output_path)

    # 2. Create it recursively (including missing parents)
    output_path.mkdir(parents=True, exist_ok=True)

    converter = A2D2TarDatasetConverter(tar_path)

    groups = converter.shuffle_and_group_pngs()

    # here possible truncation for testing

    for idx, group in enumerate(groups):
        frames = converter.proceed_one_bunch(group)
        # Here you would save 'frames' to 'output_path' as needed
        converter.save_bunch(frames, output_path, idx)
