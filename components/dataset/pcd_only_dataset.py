import bisect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from components.utils.logger import logger

DEFAULT_PREFETCH_THREAD = 12


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
    """Load point clouds recursively from one filesystem folder."""

    def __init__(
        self,
        root_dir: str,
        transform: Callable | None = None,
        prefetch_threads: int = DEFAULT_PREFETCH_THREAD,
    ):
        self._transform = transform
        self._prefetch_threads = prefetch_threads
        self._pool: ThreadPoolExecutor | None = None
        self.pcd_paths = sorted(
            path for path in Path(root_dir).rglob("*") if path.is_file() and path.suffix.lower() == ".pcd"
        )
        logger.info(f"Collected {len(self.pcd_paths)} PCD files from {root_dir}")

    def __len__(self):
        return len(self.pcd_paths)

    def _load_one(self, idx: int):
        from pypcd4 import PointCloud

        point_cloud = PointCloud.from_path(str(self.pcd_paths[idx]))
        data = point_cloud.numpy()
        if self._transform:
            data = self._transform(data)
        return data

    def __getitem__(self, idx):
        return self._load_one(idx)


class PcdOnlyDataset(torch.utils.data.Dataset):
    """Combine point-cloud files from multiple folders into one dataset.

    Files are expected to share the same fields. If point counts differ, use a
    custom DataLoader collate function to pad or otherwise combine the arrays.
    """

    def __init__(
        self,
        root_dirs: list[str],
        transform: Callable | None = None,
        prefetch_threads: int = DEFAULT_PREFETCH_THREAD,
    ):
        self.datasets: list[torch.utils.data.Dataset] = []
        for root_dir in root_dirs:
            path = Path(root_dir)
            if not path.is_dir():
                raise ValueError(f"PCD dataset path is not a directory: {root_dir}")
            self.datasets.append(PcdOnlyFolderDataset(root_dir, transform, prefetch_threads))

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
