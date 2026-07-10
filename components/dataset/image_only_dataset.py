from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

from components.utils.logger import logger


class ImageOnlyDataset(torch.utils.data.Dataset):
    """Dataset for images without annotation."""

    def __init__(self, root_dirs: list[str], transform: Optional[Callable] = None):
        """Initialize the dataset with the root directory and transformation."""
        self._transform = transform

        glob_patterns = (
            "*.jpg",
            "*.jpeg",
            "*.png",
        )
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


if __name__ == "__main__":
    import sys

    from components.utils.logger import configure_logger, logger

    if len(sys.argv) > 1:
        configure_logger("explore image-only-dataset")
        # Try to load the datasets with paths
        concated_paths = sys.argv[1]
        root_dirs = concated_paths.split(",")
        _ = ImageOnlyDataset(root_dirs=root_dirs, transform=None)
