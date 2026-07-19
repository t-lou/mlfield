import io
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

from components.utils.logger import logger

# Supported image file extensions for the dataset
EXTS = (".jpg", ".jpeg", ".png")


class ImageOnlyFolderDataset(torch.utils.data.Dataset):
    """Dataset for images without annotation."""

    def __init__(self, root_dirs: list[str], transform: Optional[Callable] = None):
        """Initialize the dataset with the root directory and transformation."""
        self._transform = transform

        glob_patterns = tuple(f"*{ext}" for ext in EXTS)
        self.image_paths = []
        for root_dir in root_dirs:
            for pattern in glob_patterns:
                self.image_paths.extend(Path(root_dir).rglob(pattern))

            logger.info(f"Collected {len(self.image_paths)} images until {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get an item from the dataset at the specified index."""
        img_path = self.image_paths[idx]
        with Image.open(img_path) as image:
            data = image.convert("RGB")
            if self._transform:
                data = self._transform(data)
        return data


class ImageOnlyZipDataset(torch.utils.data.Dataset):
    """Dataset for images without annotation, backed by one or more zip archives.

    Works with mixed ZIP_STORED / ZIP_DEFLATED members transparently.
    Defines __getitems__ so DataLoader fetches a whole batch via a thread
    pool per worker, overlapping slow filesystem reads and decompression.
    """

    def __init__(
        self,
        zip_paths: list[str],
        transform: Optional[Callable] = None,
        prefetch_threads: int = 8,
    ):
        self._transform = transform
        self._zip_paths = zip_paths
        self._prefetch_threads = prefetch_threads

        # Per-worker-process state, created lazily (see _get_zip / _get_pool)
        self._zip_handles: dict[int, zipfile.ZipFile] = {}
        self._pool: Optional[ThreadPoolExecutor] = None

        self.index: list[tuple[int, str]] = []
        for zi, zp in enumerate(zip_paths):
            with zipfile.ZipFile(zp) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(EXTS)]
                self.index.extend((zi, n) for n in names)
            logger.info(f"Indexed {len(names)} images from {zp}")

    def __len__(self):
        return len(self.index)

    def _get_zip(self, zip_idx: int) -> zipfile.ZipFile:
        # Lazily opened per worker process/thread-safe-enough for read();
        # each worker gets its own handle, avoiding cross-process fd sharing.
        if zip_idx not in self._zip_handles:
            self._zip_handles[zip_idx] = zipfile.ZipFile(self._zip_paths[zip_idx])
        return self._zip_handles[zip_idx]

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self._prefetch_threads)
        return self._pool

    def _load_one(self, idx: int):
        zip_idx, name = self.index[idx]
        zf = self._get_zip(zip_idx)
        raw = zf.read(name)  # transparently handles Stored or Deflate
        with Image.open(io.BytesIO(raw)) as image:
            data = image.convert("RGB")
        if self._transform:
            data = self._transform(data)
        return data

    def __getitem__(self, idx):
        # Still supported for compatibility (e.g. with samplers that call it directly)
        return self._load_one(idx)

    def __getitems__(self, indices: list[int]):
        # Called by DataLoader once per batch -- fetch+decompress concurrently
        pool = self._get_pool()
        return list(pool.map(self._load_one, indices))


class ImageOnlyTarDataset(torch.utils.data.Dataset):
    """Dataset for images without annotation, backed by one or more tar archives.

    Use uncompressed .tar (not .tar.gz) so members can be read via direct
    seek+read without a streaming decompressor.
    """

    def __init__(self, tar_paths: list[str], transform: Optional[Callable] = None):
        self._transform = transform
        self._tar_paths = tar_paths
        self._file_handles: dict[int, "io.BufferedReader"] = {}

        # (tar_idx, data_offset, size) built once via a single sequential pass
        self.index: list[tuple[int, int, int]] = []
        for ti, tp in enumerate(tar_paths):
            with tarfile.open(tp, "r:") as tf:  # "r:" = uncompressed only
                for member in tf:
                    if member.isfile() and member.name.lower().endswith(EXTS):
                        self.index.append((ti, member.offset_data, member.size))

    def __len__(self):
        return len(self.index)

    def _get_file(self, tar_idx: int):
        if tar_idx not in self._file_handles:
            self._file_handles[tar_idx] = open(self._tar_paths[tar_idx], "rb")
        return self._file_handles[tar_idx]

    def __getitem__(self, idx):
        tar_idx, offset, size = self.index[idx]
        f = self._get_file(tar_idx)
        f.seek(offset)
        raw = f.read(size)
        with Image.open(io.BytesIO(raw)) as image:
            data = image.convert("RGB")
        if self._transform:
            data = self._transform(data)
        return data


class ImageOnlyDataset(torch.utils.data.Dataset):
    """Container to generalize the datase.

    For each of the input path, use either
    - Folder Dataset
    - Tar Dataset
    - Zip Dataset

    Based on whether the it is a folder or a package.
    """

    def __init__(self, paths: list[str], transform: Optional[Callable] = None):
        self._transform = transform
        self.datasets: list[torch.utils.data.Dataset] = []
        for path in paths:
            p = Path(path)
            if p.is_dir():
                self.datasets.append(ImageOnlyFolderDataset([path], transform))
            elif p.suffix == ".zip":
                self.datasets.append(ImageOnlyZipDataset([path], transform))
            elif p.suffix == ".tar":
                self.datasets.append(ImageOnlyTarDataset([path], transform))
            else:
                raise ValueError(f"Unsupported dataset path: {path}")

        # Concatenate all datasets into a single dataset
        self.dataset = torch.utils.data.ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    import sys

    from components.utils.logger import configure_logger, logger

    if len(sys.argv) > 1:
        configure_logger("explore image-only-dataset")
        # Try to load the datasets with paths
        concated_paths = sys.argv[1]
        root_dirs = concated_paths.split(",")
        _ = ImageOnlyDataset(root_dirs=root_dirs, transform=None)
