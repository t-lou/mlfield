import io
import logging
import random
import sys
import tarfile
from pathlib import PurePosixPath
import json

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

    def list_timestamps(self):
        """
        Find all timestamps under:
            camera_lidar_semantic_bboxes/<timestamp>/
        """
        timestamps = set()

        for member in self.tar.getmembers():
            p = PurePosixPath(member.name)

            # Expect: camera_lidar_semantic_bboxes/<timestamp>/...
            if len(p.parts) >= 2 and p.parts[0] == self._dataset_type:
                timestamps.add(p.parts[1])

        return sorted(timestamps)

    def find_fc_files(self):
        fc_frames = []

        for member in self.tar.getmembers():
            p = PurePosixPath(member.name)

            if len(fc_frames) > 300:
                print("Shortened for testing purposes.")
                break  # shorten for testing, TODO remove

            if len(p.parts) >= 4:
                sensor_type, cam_name = p.parts[2], p.parts[3]

                if (
                    sensor_type == self._root_parsing
                    and cam_name == self._name
                    and p.parts[-1].lower().endswith(".png")
                ):
                    fc_frames.append(member.name)

        return fc_frames

    def load_lidar(self, path: PurePosixPath) -> np.ndarray:
        # Convert PurePosixPath → string for tarfile
        path_str = str(path)

        # Read raw bytes from tar
        fileobj = self.tar.extractfile(path_str)
        if fileobj is None:
            raise FileNotFoundError(f"LIDAR file {path_str} not found in tar")

        # Load npz from in-memory buffer
        data = np.load(io.BytesIO(fileobj.read()))

        points = data["points"]
        reflectance = data["reflectance"][:, None]
        timestamp = data["timestamp"][:, None]

        arr = np.concatenate([points, reflectance, timestamp], axis=1)
        logging.debug(f"Loaded LIDAR from {path_str} with shape {arr.shape}")

        # Pad or truncate to fixed size
        num = min(arr.shape[0], self._num_lidar_points)
        padded = np.zeros((self._num_lidar_points, arr.shape[1]), dtype=arr.dtype)
        padded[:num] = arr[:num]

        return padded

    def load_img(self, path: PurePosixPath) -> np.ndarray:
        path_str = str(path)

        fileobj = self.tar.extractfile(path_str)
        if fileobj is None:
            raise FileNotFoundError(f"Camera file {path_str} not found in tar")

        img = Image.open(io.BytesIO(fileobj.read())).convert("RGB")

        # Convert PIL → NumPy
        return np.array(img)

    def load_boxes(self, path: PurePosixPath) -> np.ndarray:
        path_str = str(path)

        # Read raw bytes from tar
        fileobj = self.tar.extractfile(path_str)
        if fileobj is None:
            raise FileNotFoundError(f"3D box file {path_str} not found in tar")

        # Load JSON from in-memory bytes
        data = json.load(io.BytesIO(fileobj.read()))

        boxes = []

        # data is a dict: {"box_0": {...}, "box_1": {...}, ...}
        for obj in data.values():
            center = obj["center"]  # [x, y, z]
            size = obj["size"]  # [dx, dy, dz]
            yaw = obj.get("rot_angle", 0.0)  # fallback if missing

            # center + size + [yaw] → 7 numbers
            boxes.append(center + size + [yaw])

        # Pad or truncate to fixed size
        if len(boxes) > self._num_gt_boxes:
            logging.warning(f"Truncating {len(boxes)} boxes to {self._num_gt_boxes} boxes")
            boxes = boxes[: self._num_gt_boxes]
        else:
            # pad with zeros
            boxes += [[0.0] * 7] * (self._num_gt_boxes - len(boxes))

        # If no boxes at all
        if not boxes:
            return np.zeros((0, 7), dtype=np.float32)

        return np.array(boxes, dtype=np.float32)

    def shuffle_and_group_pngs(self):
        """
        Shuffle and group the front center PNG files into groups of given size.
        """
        fc_frames = self.find_fc_files()
        random.shuffle(fc_frames)

        grouped = [fc_frames[i : i + self._group_size] for i in range(0, len(fc_frames), self._group_size)]
        if len(grouped[-1]) < self._group_size:
            grouped = grouped[:-1]  # drop last incomplete group

        return grouped

    def proceed_one_frame(self, fc_path: str) -> dict:
        parts = PurePosixPath(fc_path).parts

        def _conv_lidar() -> PurePosixPath:
            parts_lidar = list(parts)
            parts_lidar[1:] = [part.replace("camera", "lidar") for part in parts_lidar[1:]]
            return PurePosixPath(*parts_lidar).with_suffix(".npz")

        def _conv_semantic() -> PurePosixPath:
            parts_semantic = list(parts)
            parts_semantic[1:] = [part.replace("camera", "label") for part in parts_semantic[1:]]
            return PurePosixPath(*parts_semantic).with_suffix(".png")

        def _conv_bboxes() -> PurePosixPath:
            parts_bboxes = list(parts)
            parts_bboxes[1:] = [part.replace("camera", "label3D") for part in parts_bboxes[1:]]
            return PurePosixPath(*parts_bboxes).with_suffix(".json")

        paths = {
            "camera": fc_path,
            "lidar": _conv_lidar(),
            "semantics": _conv_semantic(),
            "gt_boxes": _conv_bboxes(),
        }

        for path in paths.values():
            try:
                self.tar.getmember(str(path))  # <-- FIX HERE
            except KeyError:
                raise FileNotFoundError(f"File {path} not found in tar archive.")

        points = self.load_lidar(paths["lidar"])
        cam = self.load_img(paths["camera"])
        semantics = self.load_img(paths["semantics"])
        boxes = self.load_boxes(paths["gt_boxes"])

        data = {
            "points": points,
            "camera": cam,
            "semantics": semantics,
            "gt_boxes": boxes,
        }
        return data

    def proceed_one_bunch(self, fc_paths: list[str]):
        assert len(fc_paths) == self._group_size, (self._group_size, len(fc_paths))
        assert len(fc_paths) == self._group_size, "Input group size does not match expected group size."
        for fc_path in fc_paths:
            names = self.proceed_one_frame(fc_path)
            _ = names  # TODO: process/store the data as needed


if __name__ == "__main__":
    tar_path = sys.argv[1]
    converter = A2D2TarDatasetConverter(tar_path)

    groups = converter.shuffle_and_group_pngs()
    converter.proceed_one_bunch(groups[0])
