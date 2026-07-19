import bisect
import io
import tarfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

from components.utils.logger import logger

# Supported image file extensions for the dataset
EXTS = (".jpg", ".jpeg", ".png")

DEFAULT_PREFETCH_THREAD = 12


class _ThreadedBatchMixin:
    """Adds a threaded __getitems__ on top of a subclass-provided _load_one.

    Also makes the dataset safe to pickle (spawn) or fork: thread pool and
    any per-thread handles are dropped from state and recreated lazily.
    """

    _prefetch_threads: int = DEFAULT_PREFETCH_THREAD

    def _get_pool(self) -> ThreadPoolExecutor:
        if getattr(self, "_pool", None) is None:
            self._pool = ThreadPoolExecutor(max_workers=self._prefetch_threads)
        return self._pool

    def __getitems__(self, indices: list[int]):
        pool = self._get_pool()
        return list(pool.map(self._load_one, indices))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pool"] = None
        state["_local"] = None  # subclasses store per-thread handles here
        return state


class ImageOnlyFolderDataset(_ThreadedBatchMixin, torch.utils.data.Dataset):
    """Dataset for images without annotation, backed by a plain folder."""

    def __init__(
        self,
        root_dirs: list[str],
        transform: Optional[Callable] = None,
        prefetch_threads: int = DEFAULT_PREFETCH_THREAD,
    ):
        self._transform = transform
        self._prefetch_threads = prefetch_threads
        self._pool: Optional[ThreadPoolExecutor] = None

        glob_patterns = tuple(f"*{ext}" for ext in EXTS)
        self.image_paths = []
        for root_dir in root_dirs:
            for pattern in glob_patterns:
                self.image_paths.extend(Path(root_dir).rglob(pattern))
            logger.info(f"Collected {len(self.image_paths)} images until {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_one(self, idx: int):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as image:
            data = image.convert("RGB")
        if self._transform:
            data = self._transform(data)
        return data

    def __getitem__(self, idx):
        return self._load_one(idx)


class ImageOnlyZipDataset(_ThreadedBatchMixin, torch.utils.data.Dataset):
    """Dataset for images without annotation, backed by one or more zip archives.

    Transparently supports ZIP_STORED and any ZIP_DEFLATED compression level
    (Defl:N/X/S all decode the same way -- level only matters when writing).
    Each thread opens its own ZipFile handle onto the same path, so reads
    across threads run truly concurrently instead of serializing on a shared
    file object -- important when the underlying filesystem is slow.
    """

    def __init__(
        self,
        zip_paths: list[str],
        transform: Optional[Callable] = None,
        prefetch_threads: int = DEFAULT_PREFETCH_THREAD,
    ):
        self._transform = transform
        self._zip_paths = zip_paths
        self._prefetch_threads = prefetch_threads
        self._pool: Optional[ThreadPoolExecutor] = None
        self._local: Optional[threading.local] = None

        self.index: list[tuple[int, str]] = []
        for zi, zp in enumerate(zip_paths):
            with zipfile.ZipFile(zp) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(EXTS)]
                methods = {info.compress_type for info in zf.infolist() if info.filename in names}
                unsupported = methods - {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                if unsupported:
                    raise ValueError(f"{zp} uses unsupported compression: {unsupported}")
                self.index.extend((zi, n) for n in names)
            logger.info(f"Indexed {len(names)} images from {zp}")

    def __len__(self):
        return len(self.index)

    def _get_local(self) -> threading.local:
        if self._local is None:
            self._local = threading.local()
        return self._local

    def _get_zip(self, zip_idx: int) -> zipfile.ZipFile:
        local = self._get_local()
        handles = getattr(local, "handles", None)
        if handles is None:
            handles = {}
            local.handles = handles
        if zip_idx not in handles:
            handles[zip_idx] = zipfile.ZipFile(self._zip_paths[zip_idx])
        return handles[zip_idx]

    def _load_one(self, idx: int):
        zip_idx, name = self.index[idx]
        zf = self._get_zip(zip_idx)
        raw = zf.read(name)  # handles Stored or Deflate (any level) transparently
        with Image.open(io.BytesIO(raw)) as image:
            data = image.convert("RGB")
        if self._transform:
            data = self._transform(data)
        return data

    def __getitem__(self, idx):
        return self._load_one(idx)


class ImageOnlyTarDataset(_ThreadedBatchMixin, torch.utils.data.Dataset):
    """Dataset for images without annotation, backed by one or more tar archives.

    Requires uncompressed .tar so members can be read via direct seek+read.
    Each thread opens its own file handle onto the same path -- sharing one
    handle across threads for raw seek()+read() is an actual data race, not
    just a performance issue, so this is required for correctness, not just speed.
    """

    def __init__(
        self,
        tar_paths: list[str],
        transform: Optional[Callable] = None,
        prefetch_threads: int = DEFAULT_PREFETCH_THREAD,
    ):
        self._transform = transform
        self._tar_paths = tar_paths
        self._prefetch_threads = prefetch_threads
        self._pool: Optional[ThreadPoolExecutor] = None
        self._local: Optional[threading.local] = None

        self.index: list[tuple[int, int, int]] = []
        for ti, tp in enumerate(tar_paths):
            with tarfile.open(tp, "r:") as tf:  # "r:" rejects compressed tars
                count = 0
                for member in tf:
                    if member.isfile() and member.name.lower().endswith(EXTS):
                        self.index.append((ti, member.offset_data, member.size))
                        count += 1
            logger.info(f"Indexed {count} images from {tp}")

    def __len__(self):
        return len(self.index)

    def _get_local(self) -> threading.local:
        if self._local is None:
            self._local = threading.local()
        return self._local

    def _get_file(self, tar_idx: int):
        local = self._get_local()
        handles = getattr(local, "handles", None)
        if handles is None:
            handles = {}
            local.handles = handles
        if tar_idx not in handles:
            handles[tar_idx] = open(self._tar_paths[tar_idx], "rb")
        return handles[tar_idx]

    def _load_one(self, idx: int):
        tar_idx, offset, size = self.index[idx]
        f = self._get_file(tar_idx)  # per-thread handle: no seek/read race
        f.seek(offset)
        raw = f.read(size)
        with Image.open(io.BytesIO(raw)) as image:
            data = image.convert("RGB")
        if self._transform:
            data = self._transform(data)
        return data

    def __getitem__(self, idx):
        return self._load_one(idx)


class ImageOnlyDataset(torch.utils.data.Dataset):
    """Container generalizing over Folder / Tar / Zip sources.

    Defines __getitems__ so batched fetches actually reach the underlying
    threaded datasets -- without this, DataLoader silently falls back to
    one __getitem__ call per index and none of the threading below helps.
    """

    def __init__(
        self,
        root_dirs: list[str],
        transform: Optional[Callable] = None,
        prefetch_threads: int = DEFAULT_PREFETCH_THREAD,
    ):
        self._transform = transform
        self.datasets: list[torch.utils.data.Dataset] = []
        for path in root_dirs:
            p = Path(path)
            if p.is_dir():
                self.datasets.append(ImageOnlyFolderDataset([path], transform, prefetch_threads))
            elif p.suffix == ".zip":
                self.datasets.append(ImageOnlyZipDataset([path], transform, prefetch_threads))
            elif p.suffix == ".tar":
                self.datasets.append(ImageOnlyTarDataset([path], transform, prefetch_threads))
            else:
                raise ValueError(f"Unsupported dataset path: {path}")

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
            local_indices = [p[0] for p in pairs]
            items = sub_dataset.__getitems__(local_indices)  # every sub-type now has one
            for (_, pos), item in zip(pairs, items):
                results[pos] = item
        return results


if __name__ == "__main__":
    import sys

    from components.utils.logger import configure_logger, logger

    if len(sys.argv) > 1:
        configure_logger("explore image-only-dataset")
        # Try to load the datasets with paths
        concated_paths = sys.argv[1]
        root_dirs = concated_paths.split(",")
        _ = ImageOnlyDataset(root_dirs=root_dirs, transform=None)
