from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

from components.utils.logger import logger


class ImageOnlyDataset(torch.utils.data.Dataset):
    """Dataset for images without annotation."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        """Initialize the dataset with the root directory and transformation."""
        self.root_dir = root_dir
        self._transform = transform

        glob_patterns = (
            "*.jpg",
            "*.jpeg",
            "*.png",
        )
        self.image_paths = []
        for pattern in glob_patterns:
            self.image_paths.extend(Path(root_dir).rglob(pattern))

        logger.info(f"Found {len(self.image_paths)} images in {root_dir}")

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
