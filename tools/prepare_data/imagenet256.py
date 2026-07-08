from pathlib import Path

DATASET_NAME = "dimensi0n/imagenet-256"

PATH = "data/kaggle/imagenet"


def _download() -> None:
    # Download to PATH/train and create PATH/val as dummy
    if not Path(PATH).exists():
        Path(PATH).mkdir(parents=True, exist_ok=True)
    if not Path(PATH, "train").exists():
        import kaggle

        kaggle.api.dataset_download_files(DATASET_NAME, path=f"{PATH}/train", unzip=True)
    if not Path(PATH, "val").exists():
        Path(PATH, "val").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    _download()
