import re
from pathlib import Path


def clean_indexed_files(folder: str, n: int | None, confirmed: bool = False):
    # Safety check
    if n is None or n < 2:
        print("n is not provided or < 2, exiting without changes.")
        return

    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Folder does not exist: {folder}")
        return

    # Regex to capture trailing integer index
    index_pattern = re.compile(r"(\d+)$")

    indexed_files = []

    # Collect all files with an index
    for f in folder_path.iterdir():
        if f.is_file():
            name = f.stem  # remove extension
            match = index_pattern.search(name)
            if match:
                idx = int(match.group(1))
                indexed_files.append((idx, f))

    if not indexed_files:
        print("No indexed files found.")
        return

    # Determine which indices to keep
    indices = [idx for idx, _ in indexed_files]
    max_index = max(indices)

    keep_indices = {0, max_index}
    keep_indices.update(idx for idx in indices if idx % n == 0)

    if not confirmed:
        print(f"Only checking, will keep: {keep_indices}")
        return

    # Delete files not in keep set
    for idx, f in indexed_files:
        if idx not in keep_indices:
            print(f"Deleting {f}")
            f.unlink()

    print("Cleanup complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean indexed files in a folder, keeping every n-th file.")
    parser.add_argument("folder", type=str, help="Path to the folder containing indexed files.")
    parser.add_argument("n", type=int, help="Keep every n-th file (n >= 2).")
    parser.add_argument("--confirm", action="store_true", help="Confirm deletion of files.")

    args = parser.parse_args()

    clean_indexed_files(args.folder, args.n, confirmed=args.confirm)
