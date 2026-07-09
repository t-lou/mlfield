import zipfile
from pathlib import Path


def _download_kaggle_dataset(dataset_slug: str, dest_dir: Path) -> None:
    """
    Download and extract a Kaggle dataset.

    Requires kaggle package and authentication credentials at ~/.kaggle/kaggle.json
    Automatically handles nested zip files in extracted archives.

    Args:
        dataset_slug: Kaggle dataset identifier (format: 'owner/dataset')
        dest_dir: Destination directory for downloaded files

    Raises:
        RuntimeError: If kaggle package is missing or authentication fails

    Example:
        >>> _download_kaggle_dataset('awsaf49/coco-2017-dataset', Path('./data/coco'))

    Improvement: Consider adding:
        - Resumable downloads for large files
        - Download progress bar with tqdm
        - Checksum verification after download
        - Retry logic for network failures
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Kaggle support requires the 'kaggle' package. Install with: pip install kaggle") from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local auth setup
        raise RuntimeError(
            "Kaggle authentication failed. Place credentials at ~/.kaggle/kaggle.json "
            "or set KAGGLE_USERNAME and KAGGLE_KEY."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True)

    # Some Kaggle datasets contain zip files inside the first extracted directory.
    for zip_path in dest_dir.rglob("*.zip"):
        extract_dir = zip_path.parent
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)


def argparse():
    """Parse command-line arguments for downloading a Kaggle dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Download and extract a Kaggle dataset.")
    parser.add_argument("dataset_slug", type=str, help="Kaggle dataset identifier (format: 'owner/dataset')")
    parser.add_argument("dest_dir", type=Path, help="Destination directory for downloaded files")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparse()
    _download_kaggle_dataset(args.dataset_slug, args.dest_dir)
