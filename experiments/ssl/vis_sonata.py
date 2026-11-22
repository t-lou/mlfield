"""Visualize PTv3/Sonata features on a PCD or PLY point cloud.

Examples:
    python -m experiments.ssl.vis_sonata sonata_teacher_backbone.pt cloud.pcd
    python -m experiments.ssl.vis_sonata sonata_teacher_backbone.onnx cloud.ply
    python -m experiments.ssl.vis_sonata sonata_full.onnx cloud.pcd --output unmask_logits
    python -m experiments.ssl.vis_sonata model.pt cloud.pcd --component 12 --screenshot features.png

Non-standard PT checkpoints need the Sonata config JSON used during training via ``--cfg``.
ONNX exports use their fixed point count; points are deterministically sampled or repeated
when the source cloud has a different number of points. Their traced lattice is retained by
mapping points in spatial Z-order; use a ``.pt`` checkpoint for native input geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from components.dataset.pcd_only_dataset import PcdOnlyFolderDataset
from components.mmperc.encoder.point_transformer_v3 import PointTransformerV3Config, z_order_encode
from components.utils.device import get_device
from experiments.ssl.sonata import DEFAULT_IGNORED_FIELDS, SonataBackbone, SonataConfig, upcast


def _checkpoint_backbone_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: expected a checkpoint dictionary or state dict")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError(f"{path}: checkpoint does not contain a state dict")

    prefixes = ("teacher.backbone.", "module.teacher.backbone.", "backbone.")
    for prefix in prefixes:
        selected = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if selected:
            return selected
    if "embed_conv.weight" in state:
        return state
    raise ValueError(f"{path}: could not find PTv3 backbone weights in the checkpoint")


def _read_config(config_path: Path) -> SonataConfig:
    with config_path.open(encoding="utf-8") as file:
        values = json.load(file)
    backbone = values.get("backbone")
    if isinstance(backbone, dict):
        values["backbone"] = PointTransformerV3Config(**backbone)
    return SonataConfig(**values)


def _checkpoint_config(state: dict[str, torch.Tensor], config_path: Path | None) -> SonataConfig:
    if config_path is not None:
        return _read_config(config_path)

    channels = [int(state["embed_norm.weight"].numel())]
    pool_ids = sorted(
        {int(match.group(1)) for key in state if (match := re.match(r"pools\.(\d+)\.norm\.weight$", key))}
    )
    channels.extend(int(state[f"pools.{idx}.norm.weight"].numel()) for idx in pool_ids)
    stage_ids = sorted({int(match.group(1)) for key in state if (match := re.match(r"enc\.(\d+)\.\d+\.", key))})
    depths = [
        len({int(match.group(1)) for key in state if (match := re.match(rf"enc\.{stage}\.(\d+)\.", key))})
        for stage in stage_ids
    ]
    in_channels = int(state["embed_conv.weight"].shape[1])
    kernel_volume = int(state["embed_conv.weight"].shape[0])
    stem_kernel = round(kernel_volume ** (1 / 3))
    if stem_kernel**3 != kernel_volume or len(channels) != len(depths) or stage_ids != list(range(len(channels))):
        raise ValueError("Could not infer the PTv3 architecture from this checkpoint; provide its config with --cfg")

    default_channels = (48, 96, 192, 384, 512)
    default_depths = (3, 3, 3, 12, 3)
    small_channels = (16, 32, 64, 128)
    small_depths = (1, 1, 1, 1)
    if tuple(channels) == default_channels and tuple(depths) == default_depths and stem_kernel == 5:
        backbone = replace(PointTransformerV3Config(), in_channels=in_channels)
    elif tuple(channels) == small_channels and tuple(depths) == small_depths and stem_kernel == 3:
        backbone = PointTransformerV3Config(
            in_channels=in_channels,
            order=("z",),
            stride=(2, 2, 2),
            enc_depths=small_depths,
            enc_channels=small_channels,
            enc_num_head=(2, 4, 8, 16),
            enc_patch_size=(256,) * 4,
            stem_kernel=3,
            enc_mode=True,
        )
    else:
        raise ValueError(
            "Could not infer this checkpoint's attention/order settings; pass the training config JSON with --cfg"
        )
    return SonataConfig(backbone=backbone)


def _read_ascii_ply_header(file: TextIO, path: Path) -> tuple[int, tuple[str, ...]]:
    if file.readline().strip() != "ply":
        raise ValueError(f"{path}: invalid PLY header")
    header: list[list[str]] = []
    complete = False
    for line in file:
        parts = line.split()
        if parts and parts[0] == "end_header":
            complete = True
            break
        if parts and parts[0] not in {"comment", "obj_info"}:
            header.append(parts)
    if not complete:
        raise ValueError(f"{path}: incomplete PLY header")
    return _parse_ascii_ply_header(header, path)


def _parse_ascii_ply_header(header: list[list[str]], path: Path) -> tuple[int, tuple[str, ...]]:
    format_parts = next((parts for parts in header if parts[0] == "format"), [])
    vertex_indexes = [idx for idx, parts in enumerate(header) if parts[0] == "element" and parts[1] == "vertex"]
    if format_parts[1:] != ["ascii", "1.0"] or len(vertex_indexes) != 1:
        raise ValueError(f"{path}: ASCII fallback requires a vertex element and format ascii 1.0")
    vertex_index = vertex_indexes[0]
    if any(parts[0] == "element" for parts in header[:vertex_index]):
        raise ValueError(f"{path}: ASCII fallback requires the vertex element first")
    vertex_count = int(header[vertex_index][2])
    next_element = next(
        (idx for idx in range(vertex_index + 1, len(header)) if header[idx][0] == "element"),
        len(header),
    )
    property_lines = [parts for parts in header[vertex_index + 1 : next_element] if parts[0] == "property"]
    if any(len(parts) != 3 for parts in property_lines):
        raise ValueError(f"{path}: ASCII fallback does not support list-valued vertex properties")
    fields = tuple(parts[2] for parts in property_lines)
    if not fields:
        raise ValueError(f"{path}: PLY vertex element has no scalar properties")
    return vertex_count, fields


def _load_ascii_ply(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    with path.open(encoding="ascii") as file:
        vertex_count, fields = _read_ascii_ply_header(file, path)
        if vertex_count == 0:
            return np.empty((0, len(fields)), dtype=np.float32), fields
        try:
            array = np.loadtxt(file, dtype=np.float32, max_rows=vertex_count, usecols=range(len(fields)), ndmin=2)
        except ValueError as exc:
            raise ValueError(f"{path}: could not decode ASCII PLY vertex properties: {exc}") from exc
        if array.shape[0] != vertex_count:
            raise ValueError(f"{path}: truncated PLY vertex data")
        return array, fields


def _prepare_cloud_array(
    path: Path,
    array: np.ndarray,
    fields: tuple[str, ...],
    coord_fields: tuple[str, str, str],
    selected_features: tuple[str, ...] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    field_to_idx = {field.lower(): idx for idx, field in enumerate(fields)}
    coord_names = tuple(name.lower() for name in coord_fields)
    missing = [name for name in coord_names if name not in field_to_idx]
    if missing:
        raise ValueError(f"{path}: missing coordinate fields {missing}; available fields: {fields}")

    values = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    xyz = values[:, [field_to_idx[name] for name in coord_names]]
    valid = torch.isfinite(xyz).all(dim=1)
    xyz = xyz[valid]
    if xyz.shape[0] == 0:
        raise ValueError(f"{path}: contains no finite XYZ coordinates")

    ignored = {name.lower() for name in DEFAULT_IGNORED_FIELDS}
    if selected_features is None:
        feature_names = tuple(
            field for field in fields if field.lower() not in coord_names and field.lower() not in ignored
        )
    else:
        feature_names = selected_features
        missing = [name for name in feature_names if name.lower() not in field_to_idx]
        if missing:
            raise ValueError(f"{path}: requested feature fields {missing} are absent; available fields: {fields}")
        if any(name.lower() in coord_names for name in feature_names):
            raise ValueError("Coordinate fields cannot also be listed as --feature-fields")

    if feature_names:
        feat = torch.stack(
            [values[:, field_to_idx[name.lower()]] for name in feature_names],
            dim=1,
        )[valid]
        feat = torch.nan_to_num(feat)
        feat = (feat - feat.mean(dim=0, keepdim=True)) / feat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        feat = feat.clamp(-10, 10)
    else:
        feat = xyz.clone()
    return xyz.numpy(), feat.numpy(), feature_names


def _load_cloud(
    path: Path,
    coord_fields: tuple[str, str, str],
    selected_features: tuple[str, ...] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not path.is_file() or path.suffix.lower() not in {".pcd", ".ply"}:
        raise ValueError(f"{path}: expected an existing .pcd or .ply file")
    dataset = PcdOnlyFolderDataset(str(path.parent), include_fields=True)
    matches = [idx for idx, cloud_path in enumerate(dataset.pcd_paths) if cloud_path.resolve() == path.resolve()]
    if not matches:
        raise FileNotFoundError(f"{path}: point cloud was not found by the PCD/PLY loader")
    try:
        array, fields = dataset[matches[0]]
    except ImportError:
        if path.suffix.lower() != ".ply":
            raise
        array, fields = _load_ascii_ply(path)
    return _prepare_cloud_array(path, array, tuple(fields), coord_fields, selected_features)


def _select_points(
    xyz: np.ndarray, feat: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("point count must be positive")
    rng = np.random.default_rng(seed)
    replace_points = len(xyz) < count
    indices = np.sort(rng.choice(len(xyz), size=count, replace=replace_points))
    return xyz[indices], feat[indices], indices


def _export_lattice(point_count: int) -> np.ndarray:
    side = math.ceil(point_count ** (1 / 3))
    point_ids = np.arange(point_count)
    return np.stack((point_ids // (side * side), (point_ids // side) % side, point_ids % side), axis=1).astype(np.int64)


def _map_to_export_lattice(xyz: np.ndarray, grid_size: float) -> np.ndarray:
    """Assign spatially ordered source points to the synthetic lattice used by ONNX export."""
    source_grid = np.floor((xyz - xyz.min(axis=0, keepdims=True)) / grid_size + 1e-4).astype(np.int64)
    source_max = source_grid.max(axis=0, keepdims=True)
    source_grid = np.divide(
        source_grid * ((1 << 16) - 1),
        source_max,
        out=np.zeros_like(source_grid, dtype=np.float64),
        where=source_max > 0,
    ).astype(np.int64)
    target_grid = _export_lattice(len(xyz))
    source_order = z_order_encode(torch.from_numpy(source_grid), 16).numpy().argsort(kind="stable")
    target_order = z_order_encode(torch.from_numpy(target_grid), 16).numpy().argsort(kind="stable")
    mapped_grid = np.empty_like(target_grid)
    mapped_grid[source_order] = target_grid[target_order]
    return mapped_grid


def output_point_map(
    grid_coord: np.ndarray,
    output_count: int,
    stride_factors: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Map output rows to input points using PTv3's sorted voxel-pooling order."""
    if output_count == len(grid_coord):
        return np.arange(len(grid_coord))

    candidates = []
    max_factor = int(grid_coord.max()) + 1
    if stride_factors is None:
        factors = []
        factor = 1
        while factor <= max_factor:
            factors.append(factor)
            factor *= 2
    else:
        factors = list(dict.fromkeys(stride_factors))
    for factor in factors:
        voxels, inverse = np.unique(grid_coord // factor, axis=0, return_inverse=True)
        if len(voxels) == output_count:
            candidates.append(inverse)
    if not candidates:
        raise ValueError(
            f"Model returned {output_count} features for {len(grid_coord)} points; "
            "could not match the output to a PTv3 voxel-pooling stage; provide --cfg for custom strides"
        )
    mapping = candidates[0]
    if any(not np.array_equal(mapping, candidate) for candidate in candidates[1:]):
        raise ValueError("Model output resolution is ambiguous; use grid coordinates matching the exported model")
    return mapping


def _run_checkpoint(
    state: dict[str, torch.Tensor],
    config: SonataConfig,
    xyz: np.ndarray,
    feat: np.ndarray,
    grid_size: float,
    device_name: str,
) -> np.ndarray:
    if feat.shape[1] != config.backbone.in_channels:
        raise ValueError(
            f"Checkpoint expects {config.backbone.in_channels} feature channels, but the cloud provides "
            f"{feat.shape[1]}; select matching --feature-fields or provide the correct --cfg"
        )
    device = get_device() if device_name == "auto" else torch.device(device_name)
    backbone = SonataBackbone(config.backbone).to(device)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()

    coordinates = torch.as_tensor(xyz, dtype=torch.float32, device=device)
    coordinates = coordinates - coordinates.mean(dim=0, keepdim=True)
    features = torch.as_tensor(feat, dtype=torch.float32, device=device)
    data = {
        "coord": coordinates,
        "feat": features,
        "grid_size": grid_size,
        "batch": torch.zeros(len(xyz), dtype=torch.long, device=device),
    }
    with torch.inference_mode():
        top = backbone(data)
        count = len(config.backbone.enc_channels)
        encoded = upcast(
            top,
            count - 1,
            coordinates,
            torch.zeros(len(xyz), dtype=torch.long, device=device),
            torch.zeros(len(xyz), dtype=torch.long, device=device),
            None,
        )
    return encoded.feat.float().cpu().numpy()


def _run_onnx(
    path: Path,
    xyz: np.ndarray,
    feat: np.ndarray,
    grid_size: float,
    requested_points: int | None,
    output_name: str,
    seed: int,
    config: SonataConfig | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX visualization requires onnxruntime in the active Python environment") from exc

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {"features", "grid_coord", "batch"}:
        raise ValueError(f"{path}: expected inputs features, grid_coord, and batch; found {tuple(inputs)}")
    point_dim = inputs["features"].shape[0]
    if point_dim is None:
        if requested_points is None:
            raise ValueError("This ONNX model has a dynamic point count; provide --num-points")
        point_count = requested_points
    else:
        point_count = int(point_dim)
        if requested_points is not None and requested_points != point_count:
            raise ValueError(f"ONNX model requires {point_count} points, but --num-points={requested_points}")

    xyz_input, feat_input, _ = _select_points(xyz, feat, point_count, seed)
    grid_coord = _map_to_export_lattice(xyz_input, grid_size)
    if feat_input.shape[1] != inputs["features"].shape[1]:
        raise ValueError(
            f"ONNX model expects {inputs['features'].shape[1]} feature channels, "
            f"but the cloud provides {feat_input.shape[1]}"
        )
    feed = {
        "features": np.ascontiguousarray(feat_input, dtype=np.float32),
        "grid_coord": np.ascontiguousarray(grid_coord),
        "batch": np.zeros(point_count, dtype=np.int64),
    }

    output_names = [item.name for item in session.get_outputs()]
    selected_output = (
        ("encoded_features" if "encoded_features" in output_names else "mask_logits")
        if output_name == "auto"
        else output_name
    )
    if selected_output not in output_names:
        raise ValueError(f"{path}: output {selected_output!r} is unavailable; outputs are {output_names}")
    print("ONNX export uses a traced cubic grid; mapping cloud points onto that grid in spatial Z-order.")
    output = np.asarray(session.run([selected_output], feed)[0])
    stride_factors = None
    if config is not None:
        backbone = config.backbone
        strides = backbone.stride or (2,) * (len(backbone.enc_depths) - 1)
        stage = (
            len(backbone.enc_depths) - 1
            if selected_output == "encoded_features"
            else len(backbone.enc_depths) - 1 - config.up_cast_level
        )
        if len(strides) != len(backbone.enc_depths) - 1 or not 0 <= stage < len(backbone.enc_depths):
            raise ValueError("The provided Sonata config has incompatible PTv3 strides or up_cast_level")
        stride_factors = (1, *np.cumprod(strides[:stage]).astype(int).tolist())
    mapping = output_point_map(grid_coord, output.shape[0], stride_factors)
    return xyz_input, output[mapping], selected_output


def _parse_fields(value: str) -> tuple[str, ...]:
    fields = tuple(field.strip() for field in value.split(",") if field.strip())
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("expected exactly three comma-separated field names")
    return fields


def _feature_rgb(features: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(features, dtype=np.float32))
    centered = values - values.mean(axis=0, keepdims=True)
    if not np.any(centered):
        return np.full((len(values), 3), 0.5, dtype=np.float32)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((len(values), 3), dtype=np.float32)
    width = min(3, left.shape[1])
    components[:, :width] = left[:, :width] * singular[:width]
    for index in range(width):
        low, high = np.percentile(components[:, index], (1, 99))
        if high > low:
            components[:, index] = (components[:, index] - low) / (high - low)
        components[:, index] = np.clip(components[:, index], 0, 1)
    return components


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {value}") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize point-wise PTv3/Sonata features.")
    parser.add_argument("model", type=Path, help="PT checkpoint (.pt) or ONNX model (.onnx)")
    parser.add_argument("pointcloud", type=Path, help="input .pcd or .ply file")
    parser.add_argument(
        "--cfg",
        type=Path,
        help="Sonata training config JSON (needed for nonstandard PT checkpoints or custom ONNX strides)",
    )
    parser.add_argument("--coord-fields", type=_parse_fields, default=("x", "y", "z"), metavar="X,Y,Z")
    parser.add_argument("--feature-fields", help="comma-separated input fields; default excludes XYZ and labels")
    parser.add_argument("--grid-size", type=_positive_float, default=0.02, help="input voxel size (default: 0.02)")
    parser.add_argument(
        "--num-points",
        type=int,
        help="PT point cap (default: 4096); ONNX models require their fixed count",
    )
    parser.add_argument(
        "--output",
        choices=("auto", "encoded_features", "mask_logits", "unmask_logits"),
        default="auto",
        help="ONNX output tensor to visualize (default: encoded features, otherwise mask logits)",
    )
    parser.add_argument("--component", type=int, help="show one feature channel with a colormap instead of PCA RGB")
    parser.add_argument("--point-size", type=_positive_float, default=3.0)
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--background", default="black")
    parser.add_argument("--screenshot", type=Path, help="save a screenshot instead of opening a window")
    parser.add_argument("--seed", type=int, default=0, help="deterministic ONNX point selection seed")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="PyTorch device for .pt checkpoints (default: best available)",
    )
    return parser.parse_args()


def _extract_features(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str, tuple[str, ...]]:
    if not args.model.is_file():
        raise FileNotFoundError(f"model file does not exist: {args.model}")
    feature_fields = None
    if args.feature_fields is not None:
        feature_fields = tuple(field.strip() for field in args.feature_fields.split(",") if field.strip())
        if not feature_fields:
            raise ValueError("--feature-fields must contain at least one field name")
    xyz, feat, names = _load_cloud(args.pointcloud, args.coord_fields, feature_fields)

    model_type = args.model.suffix.lower()
    if model_type == ".pt":
        state = _checkpoint_backbone_state(args.model)
        config = _checkpoint_config(state, args.cfg)
        point_limit = 4096 if args.num_points is None else args.num_points
        if point_limit < 1:
            raise ValueError("--num-points must be positive")
        input_xyz, input_feat, _ = _select_points(xyz, feat, min(point_limit, len(xyz)), args.seed)
        if len(input_xyz) < len(xyz):
            print(f"PT checkpoint: deterministically sampled {len(input_xyz):,} of {len(xyz):,} input points")
        features = _run_checkpoint(state, config, input_xyz, input_feat, args.grid_size, args.device)
        return input_xyz, features, "encoded_features", names
    if model_type == ".onnx":
        config = _read_config(args.cfg) if args.cfg is not None else None
        display_xyz, features, output_name = _run_onnx(
            args.model,
            xyz,
            feat,
            args.grid_size,
            args.num_points,
            args.output,
            args.seed,
            config,
        )
        return display_xyz, features, output_name, names
    raise ValueError(f"unsupported model extension {model_type!r}; expected .pt or .onnx")


def _show_cloud(args: argparse.Namespace, xyz: np.ndarray, features: np.ndarray, output_name: str) -> str:
    if len(xyz) != len(features):
        raise ValueError(f"feature/coordinate count mismatch: {len(features)} vs {len(xyz)}")
    if args.component is None:
        colors = _feature_rgb(features)
        color_name = "PCA feature RGB"
    else:
        component = args.component
        if component < 0:
            component += features.shape[1]
        if not 0 <= component < features.shape[1]:
            raise ValueError(f"component {args.component} is outside the feature width {features.shape[1]}")
        colors = features[:, component]
        color_name = f"feature channel {component}"

    import pyvista as pv

    cloud = pv.PolyData(np.asarray(xyz, dtype=np.float32))
    cloud["feature_rgb" if args.component is None else "feature_value"] = colors
    plotter = pv.Plotter(off_screen=args.screenshot is not None)
    try:
        plotter.set_background(args.background)
        if args.component is None:
            plotter.add_mesh(
                cloud,
                scalars="feature_rgb",
                rgb=True,
                point_size=args.point_size,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )
        else:
            plotter.add_mesh(
                cloud,
                scalars="feature_value",
                cmap=args.cmap,
                point_size=args.point_size,
                render_points_as_spheres=True,
                scalar_bar_args={"title": color_name},
            )
        plotter.add_axes()
        plotter.reset_camera()
        if args.screenshot is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            plotter.show(screenshot=str(args.screenshot))
            print(f"saved screenshot: {args.screenshot}")
        else:
            plotter.show(title=f"Sonata features: {args.pointcloud.name}")
    finally:
        plotter.close()
    return color_name


def main() -> int:
    args = _parse_args()
    try:
        xyz, features, output_name, names = _extract_features(args)
        color_name = _show_cloud(args, xyz, features, output_name)
        print(
            f"{args.pointcloud}: {len(xyz):,} points, {features.shape[1]} {output_name} channels, "
            f"colored by {color_name}; input fields={names}"
        )
        return 0
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
