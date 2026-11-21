#!/usr/bin/env python3
"""Random test-data generator.

Usage:
    python gen_data.py image
    python gen_data.py 3d
    python gen_data.py            # asks interactively
    python gen_data.py image --count 5 --seed 42 --output ./data/test

All attributes live in the CONFIG section below.
Requires: numpy, pillow   (pip install numpy pillow)
"""

import argparse
import hashlib
import io
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# CONFIG (all predefined here)
# ----------------------------------------------------------------------------
type PcdFieldType = Literal["F", "U", "I"]
type PcdBounds = tuple[float, float] | tuple[int, int]
type PcdField = tuple[PcdFieldType, int, PcdBounds]
type ImageFormat = Literal["png", "jpg", "random"]

OUTPUT_DIR: Path = Path("./data/gen/")
SEED: int | None = None  # int for reproducible output, None for fully random
HASH_ALGO: str = "sha1"  # filename = hex digest of file content (sha1 -> 40 chars)
DATA_COUNT: int = 200

# --- image ------------------------------------------------------------------
IMAGE_COUNT: int = DATA_COUNT
IMAGE_SIZE: tuple[int, int] = (640, 480)  # (width, height)
IMAGE_FORMAT: ImageFormat = "random"  # "png" | "jpg" | "random" (picks per file)
IMAGE_CHANNELS: Literal[1, 3] = 3  # 3 = RGB, 1 = grayscale
JPG_QUALITY: int = 90

# --- 3d / pcd ---------------------------------------------------------------
PCD_COUNT: int = DATA_COUNT  # number of files
PCD_POINTS: int = 100_000  # points per file
# name: (pcd TYPE, byte SIZE, (min, max))
#   TYPE: F = float, U = unsigned int, I = signed int
PCD_FIELDS: dict[str, PcdField] = {
    "x": ("F", 4, (-200.0, 200.0)),
    "y": ("F", 4, (-200.0, 200.0)),
    "z": ("F", 4, (-10.0, 20.0)),
    "intensity": ("F", 4, (0.0, 255.0)),
    "label": ("U", 4, (0, 20)),  # integer range, max inclusive
}
# ----------------------------------------------------------------------------


def sha_name(data: bytes) -> str:
    """Return the configured hexadecimal hash of a byte sequence.

    Args:
        data: Bytes to hash.

    Returns:
        The lowercase hexadecimal digest.

    Raises:
        ValueError: If ``HASH_ALGO`` is not a supported hashlib algorithm.
    """
    try:
        return hashlib.new(HASH_ALGO, data).hexdigest()
    except ValueError as exc:
        raise ValueError(f"Unsupported HASH_ALGO: {HASH_ALGO}") from exc


def validate_image_config() -> None:
    """Validate image-generation constants.

    Raises:
        ValueError: If an image setting is invalid.
    """
    if len(IMAGE_SIZE) != 2 or any(not isinstance(value, int) or value <= 0 for value in IMAGE_SIZE):
        raise ValueError("IMAGE_SIZE must contain two positive integers")
    if IMAGE_CHANNELS not in (1, 3):
        raise ValueError("IMAGE_CHANNELS must be 1 or 3")
    if IMAGE_FORMAT not in ("png", "jpg", "random"):
        raise ValueError('IMAGE_FORMAT must be "png", "jpg", or "random"')
    if not isinstance(JPG_QUALITY, int) or not 1 <= JPG_QUALITY <= 100:
        raise ValueError("JPG_QUALITY must be between 1 and 100")
    if SEED is not None and (not isinstance(SEED, int) or isinstance(SEED, bool)):
        raise ValueError("SEED must be an int or None")


def validate_float_bounds(name: str, lo: object, hi: object) -> None:
    """Validate bounds for a floating-point PCD field.

    Args:
        name: Field name used in validation messages.
        lo: Lower bound.
        hi: Upper bound.

    Raises:
        ValueError: If either bound is non-numeric or non-finite.
    """
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        raise ValueError(f"Float PCD field {name} bounds must be numeric")
    if not np.isfinite([lo, hi]).all():
        raise ValueError(f"Float PCD field {name} bounds must be finite")


def validate_bound_order(name: str, lo: object, hi: object) -> None:
    """Ensure numeric lower and upper bounds are ordered.

    Args:
        name: Field name used in validation messages.
        lo: Lower bound.
        hi: Upper bound.

    Raises:
        ValueError: If the lower bound exceeds the upper bound.
    """
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
        raise ValueError(f"PCD field {name} has min greater than max")


def validate_integer_bounds(name: str, field_type: str, size: int, lo: object, hi: object) -> None:
    """Validate integer PCD bounds against the configured NumPy dtype.

    Args:
        name: Field name used in validation messages.
        field_type: PCD field type, ``U`` or ``I``.
        size: Field size in bytes.
        lo: Lower bound.
        hi: Upper bound.

    Raises:
        ValueError: If bounds are not integers or exceed the dtype range.
    """
    if not isinstance(lo, int) or not isinstance(hi, int):
        raise ValueError(f"Integer PCD field {name} bounds must be integers")
    dtype = np.dtype(f"<{'u' if field_type == 'U' else 'i'}{size}")
    dtype_info = np.iinfo(dtype)
    if lo < dtype_info.min or hi > dtype_info.max:
        raise ValueError(f"PCD field {name} range does not fit {dtype} (allowed: {dtype_info.min} to {dtype_info.max})")


def validate_pcd_field(name: str, field_type: PcdFieldType, size: int, bounds: PcdBounds) -> None:
    """Validate one PCD field configuration.

    Args:
        name: PCD field name.
        field_type: PCD type identifier.
        size: Field size in bytes.
        bounds: Inclusive lower and upper bounds.

    Raises:
        ValueError: If the field name, type, size, or bounds are invalid.
    """
    if not name or any(char.isspace() for char in name):
        raise ValueError(f"Invalid PCD field name: {name!r}")

    allowed_sizes = {"F": (2, 4, 8), "U": (1, 2, 4, 8), "I": (1, 2, 4, 8)}
    if field_type not in allowed_sizes:
        raise ValueError(f"Invalid PCD field type for {name}: {field_type}")
    if size not in allowed_sizes[field_type]:
        raise ValueError(f"Unsupported size {size} bytes for PCD field {name}")
    if len(bounds) != 2:
        raise ValueError(f"PCD field {name} must define (min, max)")

    lo, hi = bounds
    validate_bound_order(name, lo, hi)
    if field_type == "F":
        validate_float_bounds(name, lo, hi)
        return
    validate_integer_bounds(name, field_type, size, lo, hi)


def validate_config() -> None:
    """Validate all image and PCD generator configuration.

    Raises:
        ValueError: If a hash algorithm or generator setting is invalid.
    """
    validate_image_config()

    try:
        hashlib.new(HASH_ALGO)
    except ValueError as exc:
        raise ValueError(f"Unsupported HASH_ALGO: {HASH_ALGO}") from exc

    counts = (("IMAGE_COUNT", IMAGE_COUNT), ("PCD_COUNT", PCD_COUNT), ("PCD_POINTS", PCD_POINTS))
    for name, value in counts:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not PCD_FIELDS:
        raise ValueError("PCD_FIELDS must contain at least one field")

    for name, (field_type, size, bounds) in PCD_FIELDS.items():
        validate_pcd_field(name, field_type, size, bounds)


def make_image(rng: np.random.Generator) -> tuple[bytes, str]:
    """Generate one random image.

    Args:
        rng: NumPy random generator used for pixels and format selection.

    Returns:
        A tuple containing encoded image bytes and its file extension.
    """
    w, h = IMAGE_SIZE
    fmt = IMAGE_FORMAT
    if fmt == "random":
        fmt = str(rng.choice(["png", "jpg"]))
    shape = (h, w, IMAGE_CHANNELS) if IMAGE_CHANNELS == 3 else (h, w)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB" if IMAGE_CHANNELS == 3 else "L")
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=JPG_QUALITY)
    return buf.getvalue(), fmt


def make_pcd(rng: np.random.Generator) -> bytes:
    """Generate one random PCD v0.7 binary point cloud.

    Args:
        rng: NumPy random generator used for field values.

    Returns:
        Complete PCD file bytes, including the ASCII header and binary data.
    """
    n = PCD_POINTS
    dtype = []
    for name, (t, size, _) in PCD_FIELDS.items():
        kind = {"F": "f", "U": "u", "I": "i"}[t]
        dtype.append((name, f"<{kind}{size}"))
    arr = np.empty(n, dtype=dtype)
    for name, (t, size, (lo, hi)) in PCD_FIELDS.items():
        if t == "F":
            arr[name] = rng.uniform(lo, hi, n)
        else:
            dtype_name = f"<{'u' if t == 'U' else 'i'}{size}"
            arr[name] = rng.integers(lo, hi + 1, n, dtype=dtype_name)

    names = list(PCD_FIELDS)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(names)}\n"
        f"SIZE {' '.join(str(PCD_FIELDS[k][1]) for k in names)}\n"
        f"TYPE {' '.join(PCD_FIELDS[k][0] for k in names)}\n"
        f"COUNT {' '.join('1' for _ in names)}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    return header.encode("ascii") + arr.tobytes()


def main() -> None:
    """Parse CLI arguments and generate the requested test data files."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", nargs="?", choices=["image", "3d"])
    ap.add_argument("--count", type=int, help="number of files to generate")
    ap.add_argument("--seed", type=int, help="random seed")
    ap.add_argument("--output", type=Path, help="output directory")
    args = ap.parse_args()

    mode = args.mode
    if mode is None:
        mode = ""
        while mode not in ("image", "3d"):
            try:
                mode = input('Choose "image" or "3d": ').strip().lower()
            except EOFError:
                ap.error("mode is required when stdin is not interactive")

    try:
        validate_config()
    except (TypeError, ValueError) as exc:
        ap.error(str(exc))

    seed = args.seed if args.seed is not None else SEED
    output_dir = args.output if args.output is not None else OUTPUT_DIR
    count = args.count if args.count is not None else (IMAGE_COUNT if mode == "image" else PCD_COUNT)
    if count < 0:
        ap.error("--count must be non-negative")

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for _ in range(count):
        if mode == "image":
            data, ext = make_image(rng)
        else:
            data, ext = make_pcd(rng), "pcd"
        path = output_dir / f"{sha_name(data)}.{ext}"
        path.write_bytes(data)
        print(path)
        generated += 1

    print(f"Generated {generated} {mode} file{'s' if generated != 1 else ''}")


if __name__ == "__main__":
    main()
