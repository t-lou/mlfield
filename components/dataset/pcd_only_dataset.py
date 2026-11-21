import bisect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from components.utils.logger import logger

DEFAULT_PREFETCH_THREAD = 12


def _read_ply_vertex_fields(path: Path) -> tuple[str, ...]:
    fields = []
    in_vertices = False
    has_end_header = False
    with path.open("rb") as ply_file:
        if ply_file.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        for line in ply_file:
            parts = line.decode("ascii").split()
            if not parts:
                continue
            if parts[0] == "end_header":
                has_end_header = True
                break
            if parts[0] == "element" and len(parts) >= 2:
                in_vertices = parts[1] == "vertex"
            elif in_vertices and parts[0] == "property" and len(parts) == 3:
                fields.append(parts[2])

    if not has_end_header or not fields:
        raise ValueError(f"PLY file has no vertex properties or an incomplete header: {path}")
    return tuple(fields)


def _load_ply(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    import open3d as o3d

    fields = _read_ply_vertex_fields(path)
    point_cloud = o3d.t.io.read_point_cloud(str(path))
    attributes = {name: tensor.numpy() for name, tensor in point_cloud.point.items()}

    field_sources = {
        "x": ("positions", 0),
        "y": ("positions", 1),
        "z": ("positions", 2),
        "red": ("colors", 0),
        "green": ("colors", 1),
        "blue": ("colors", 2),
        "nx": ("normals", 0),
        "ny": ("normals", 1),
        "nz": ("normals", 2),
    }
    columns = []
    for field in fields:
        if field in field_sources:
            source, index = field_sources[field]
            if source not in attributes:
                raise ValueError(f"PLY file is missing the {source} attribute required by {field}: {path}")
            columns.append(attributes[source][:, index])
        elif field in attributes:
            columns.append(attributes[field].reshape(-1))
        else:
            raise ValueError(f"Open3D did not load PLY vertex property {field!r} from {path}")

    return np.column_stack(columns), fields


class _ThreadedBatchMixin:
    """Add threaded batch loading to a dataset implementing ``_load_one``."""

    _prefetch_threads: int = DEFAULT_PREFETCH_THREAD

    def _get_pool(self) -> ThreadPoolExecutor:
        if getattr(self, "_pool", None) is None:
            self._pool = ThreadPoolExecutor(max_workers=self._prefetch_threads)
        return self._pool

    def __getitems__(self, indices: list[int]):
        return list(self._get_pool().map(self._load_one, indices))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pool"] = None
        return state


class PcdOnlyFolderDataset(_ThreadedBatchMixin, torch.utils.data.Dataset):
    """Load PCD and PLY point clouds recursively from one filesystem folder."""

    def __init__(
        self,
        root_dir: str,
        transform: Callable | None = None,
        prefetch_threads: int = DEFAULT_PREFETCH_THREAD,
        include_fields: bool = False,
    ):
        self._transform = transform
        self._prefetch_threads = prefetch_threads
        self._include_fields = include_fields
        self._pool: ThreadPoolExecutor | None = None
        self.pcd_paths = sorted(
            path for path in Path(root_dir).rglob("*") if path.is_file() and path.suffix.lower() in {".pcd", ".ply"}
        )
        logger.info(f"Collected {len(self.pcd_paths)} PCD/PLY files from {root_dir}")

    def __len__(self):
        return len(self.pcd_paths)

    def _load_one(self, idx: int):
        path = self.pcd_paths[idx]
        if path.suffix.lower() == ".ply":
            data, fields = _load_ply(path)
        else:
            from pypcd4 import PointCloud

            point_cloud = PointCloud.from_path(str(path))
            data = point_cloud.numpy()
            fields = tuple(point_cloud.fields)
        if self._transform:
            data = self._transform(data)
        return (data, fields) if self._include_fields else data

    def __getitem__(self, idx):
        return self._load_one(idx)


class PcdOnlyDataset(torch.utils.data.Dataset):
    """Combine PCD and PLY point-cloud files from multiple folders into one dataset.

    By default, samples contain only the point arrays. With ``include_fields=True``,
    each sample also includes its field names so consumers can align varying schemas.
    """

    def __init__(
        self,
        root_dirs: list[str],
        transform: Callable | None = None,
        prefetch_threads: int = DEFAULT_PREFETCH_THREAD,
        include_fields: bool = False,
    ):
        self.datasets: list[torch.utils.data.Dataset] = []
        for root_dir in root_dirs:
            path = Path(root_dir)
            if not path.is_dir():
                raise ValueError(f"Point-cloud dataset path is not a directory: {root_dir}")
            self.datasets.append(PcdOnlyFolderDataset(root_dir, transform, prefetch_threads, include_fields))

        self.dataset = torch.utils.data.ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __getitems__(self, indices: list[int]):
        cumulative = self.dataset.cumulative_sizes
        grouped: dict[int, list[tuple[int, int]]] = {}
        for pos, idx in enumerate(indices):
            dataset_idx = bisect.bisect_right(cumulative, idx)
            local_idx = idx if dataset_idx == 0 else idx - cumulative[dataset_idx - 1]
            grouped.setdefault(dataset_idx, []).append((local_idx, pos))

        results: list = [None] * len(indices)
        for dataset_idx, pairs in grouped.items():
            sub_dataset = self.datasets[dataset_idx]
            local_indices = [pair[0] for pair in pairs]
            items = sub_dataset.__getitems__(local_indices)
            for (_, pos), item in zip(pairs, items):
                results[pos] = item
        return results
