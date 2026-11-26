"""Point Transformer V3 (PTv3) in pure PyTorch (encoder + decoder).

No spconv / flash-attn / torch_scatter required. Follows the official PTv3 design
(Wu et al., CVPR 2024, "Point Transformer V3: Simpler, Faster, Stronger"):

  1. Point serialization (Z-order, Hilbert and their "-trans" variants), orders shuffled
     across blocks during training.
  2. Patch attention on the serialized order (patch padding re-uses the previous patch).
  3. xCPE: sparse 3x3x3 submanifold conv -> Linear -> LayerNorm, added residually.
  4. Stem: sparse 5x5x5 submanifold conv + BN + GELU.
  5. Serialization-based grid pooling (max-pool of projected features) and a U-Net
     decoder with grid unpooling (skip projection + upsampled projection).

Submanifold sparse convs are implemented by neighbour lookup (sorted hash keys +
searchsorted) followed by gather + matmul, i.e. they are mathematically equal to
spconv.SubMConv3d (stride 1).

Optional, off by default (so the default model matches the canonical one):
  * qk_norm: LayerNorm on per-head q and k (attention-logit stabilisation, as used in
    later ViT work). Not part of the original PTv3.

Input dict:
  coord      (N, 3) float
  feat       (N, C_in) float
  grid_coord (N, 3) long   integer voxel indices (preferred; one point per voxel), or
  grid_size  float         voxel size; grid_coord = floor((coord - min_per_cloud) / grid_size)
  batch      (N,) long     or  offset (B,) long (cumulative point counts)

Output: Point (dict with attribute access). `point.feat` is (N, dec_channels[0]) in the
input point order (or the last encoder stage if enc_mode=True). When available,
`point.encoder_stages` contains copies of the point features and coordinates after
each encoder resolution stage.

High-level data flow (S = number of encoder stages, default 5)::

    input ──► stem (SubMConv 5³ + BN + GELU)
          ──► stage 0: serialize → neighbours → Block × enc_depths[0]
          ──► GridPooling ──► stage 1: serialize → neighbours → Block × enc_depths[1]
          ──► ...                                            (S - 1 pooling steps)
          ──► [decoder, unless enc_mode] for each stage from deep to shallow:
                GridUnpooling (back to the parent resolution) → Block × dec_depths[s]

Every ``Block`` is pre-norm and consists of three residual branches:
``xCPE`` (sparse conv for positional information) → serialized patch attention → MLP.

Conventions used throughout:
  * ``N``   number of points (or occupied voxels) at the current resolution.
  * ``B``   number of point clouds in the batch.
  * ``C``   channel width of the current stage.
  * ``ps``  patch size, i.e. how many consecutive points (in serialized order) attend to
    each other.
  * Sparse neighbour tables use the value ``N`` as the "missing neighbour" sentinel; a
    zero feature row is appended so that index ``N`` gathers zeros.

The ``Point`` dict carries these keys over its lifetime (created by the stage that is
named in brackets):
  coord, feat, grid_coord, batch     (_prepare / GridPooling)
  serial_order  (n_orders, N)        argsort of the serialization codes   (serialize)
  serial_inverse (n_orders, N)       inverse permutation of serial_order  (serialize)
  cloud_sizes   (B,), min_cloud      points per cloud / the smallest one  (serialize)
  layout_cache                       per-stage cache of patch layouts     (serialize)
  nbr           (N, k³)              submanifold neighbour table          (forward)
  pooling_parent, pooling_inverse    link back to the finer level         (GridPooling)
  encoder_stages                     per-stage snapshots                  (forward)

Geometry configuration:
  PointGridSpec(min_xyz, max_xyz, voxel_size_xyz)
      Describes physical input bounds and XYZ voxel fineness. ``voxel_size_xyz`` may
      be a scalar for isotropic grids or ``(vx, vy, vz)`` for anisotropic grids.
  adapt_ptv3_config_for_input(config, grid)
      Derives the smallest valid serialization depth for the grid.
  adapt_ptv3_config_for_output(config, input_grid, output_voxel_size_xyz)
      Derives stage strides for an isotropic output voxel size. Because the current
      GridPooling implementation uses one scalar stride per stage for every XYZ axis,
      anisotropic output scaling must be implemented separately before using this API.

Example::

    lidar_grid = PointGridSpec(
        min_xyz=(0.0, -48.0, -5.0),
        max_xyz=(128.0, 48.0, 3.0),
        voxel_size_xyz=(1.0, 1.0, 8.0),
    )
    config = adapt_ptv3_config_for_input(config, lidar_grid)
    config = adapt_ptv3_config_for_output(config, lidar_grid, 8.0)
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.definitions.bev_params import BevParams


@dataclass(frozen=True)
class PointTransformerV3Config:
    """Architecture and regularization settings for the PTv3 encoder.

    Stage-wise sequences (``enc_*``) have one entry per encoder stage (``S`` entries).
    Decoder sequences (``dec_*``) have ``S - 1`` entries and are indexed by the *encoder
    stage they decode into*: ``dec_channels[0]`` is the width of the finest decoder
    stage, which is also the width of the final output features.

    Attributes:
        in_channels: Number of input feature channels (``feat.shape[1]``), e.g. 6 for
            xyz + rgb.
        order: Serialization orders. Supported: ``"z"``, ``"hilbert"`` and their
            ``"-trans"`` variants, which swap the axis roles (x<->z) before encoding.
            Block ``i`` of a stage uses ``order[i % len(order)]``.
        stride: Grid-pooling stride between consecutive stages (``S - 1`` values).
            ``None`` means 2 for every transition.
        enc_depths: Number of transformer blocks per encoder stage.
        enc_channels: Channel width per encoder stage. The stem outputs
            ``enc_channels[0]``.
        enc_num_head: Attention heads per encoder stage (must divide the channel width).
        enc_patch_size: Maximum patch size per encoder stage. The effective patch size is
            clipped to the smallest cloud in the batch (see ``patch_layout``).
        dec_depths: Number of blocks per decoder stage.
        dec_channels: Channel width per decoder stage (finest stage first).
        dec_num_head: Attention heads per decoder stage.
        dec_patch_size: Maximum patch size per decoder stage.
        mlp_ratio: Hidden width of the block MLP relative to the channel width.
        qk_norm: Apply LayerNorm to per-head q and k (not part of canonical PTv3).
        attn_drop: Dropout probability on attention weights (training only).
        proj_drop: Dropout probability after the attention output projection and inside
            the MLP.
        drop_path: Maximum stochastic-depth rate. The rate grows linearly from 0 in the
            first block to this value in the last block (separately for encoder and
            decoder).
        shuffle_orders: Randomly permute the order list for each stage during training so
            successive blocks see different serialization orders. Ignored in eval mode.
        serial_depth: Bits per axis for the serialization codes. Grid coordinates must lie
            in ``[0, 2**serial_depth)``; at most 21 (int64 Morton codes). Also used as the
            hash grid bound for neighbour lookup.
        stem_kernel: Kernel size (odd) of the sparse convolution in the stem.
        enc_mode: If True, build the encoder only and return the last encoder stage.
    """

    in_channels: int = 6
    order: Sequence[str] = ("z", "z-trans", "hilbert", "hilbert-trans")
    stride: Sequence[int] | None = None
    enc_depths: Sequence[int] = (2, 2, 2, 6, 2)
    enc_channels: Sequence[int] = (32, 64, 128, 256, 512)
    enc_num_head: Sequence[int] = (2, 4, 8, 16, 32)
    enc_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024)
    dec_depths: Sequence[int] = (2, 2, 2, 2)
    dec_channels: Sequence[int] = (64, 64, 128, 256)
    dec_num_head: Sequence[int] = (4, 4, 8, 16)
    dec_patch_size: Sequence[int] = (1024, 1024, 1024, 1024)
    mlp_ratio: float = 4.0
    qk_norm: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path: float = 0.3
    shuffle_orders: bool = True
    serial_depth: int = 16
    stem_kernel: int = 5
    enc_mode: bool = False


@dataclass(frozen=True)
class PointGridSpec:
    """Physical fineness of the input voxel grid, in XYZ units."""

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    voxel_size_xyz: tuple[float, float, float] | float

    @classmethod
    def from_bev_params(cls, params: BevParams) -> "PointGridSpec":
        return cls(
            min_xyz=(params.x_range[0], params.y_range[0], params.z_range[0]),
            max_xyz=(params.x_range[1], params.y_range[1], params.z_range[1]),
            voxel_size_xyz=params.voxel_size,
        )

    def __post_init__(self) -> None:
        for name, values in (
            ("min_xyz", self.min_xyz),
            ("max_xyz", self.max_xyz),
        ):
            if len(values) != 3:
                raise ValueError(f"{name} must contain exactly three values")
        if not all(math.isfinite(float(v)) for v in (*self.min_xyz, *self.max_xyz)):
            raise ValueError("Point-grid bounds must be finite")
        if any(maximum <= minimum for minimum, maximum in zip(self.min_xyz, self.max_xyz, strict=True)):
            raise ValueError("Every max_xyz value must be greater than its min_xyz value")
        if not all(math.isfinite(v) and v > 0 for v in self.voxel_size):
            raise ValueError("voxel size values must be positive and finite")

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        if isinstance(self.voxel_size_xyz, (int, float)):
            size = float(self.voxel_size_xyz)
            return (size, size, size)
        if len(self.voxel_size_xyz) != 3:
            raise ValueError("voxel_size_xyz must be a scalar or contain exactly three values")
        return tuple(float(v) for v in self.voxel_size_xyz)

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return tuple(
            max(1, math.ceil((maximum - minimum) / size))
            for minimum, maximum, size in zip(self.min_xyz, self.max_xyz, self.voxel_size, strict=True)
        )

    @property
    def serial_depth(self) -> int:
        """Smallest Morton/Hilbert coordinate depth needed for this grid."""
        return max(1, math.ceil(math.log2(max(self.dimensions))))


@dataclass(frozen=True)
class BEVGridSpec:
    """Contract for projecting PTv3 points into a dense XY BEV map."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    height: int
    width: int

    @classmethod
    def from_bev_params(cls, params: BevParams) -> "BEVGridSpec":
        if params.backbone_stride < 1:
            raise ValueError("BEV backbone_stride must be positive")
        return cls(
            x_range=params.x_range,
            y_range=params.y_range,
            height=params.bev_h // params.backbone_stride,
            width=params.bev_w // params.backbone_stride,
        )

    def __post_init__(self) -> None:
        for name, axis_range in (("x_range", self.x_range), ("y_range", self.y_range)):
            if len(axis_range) != 2 or axis_range[1] <= axis_range[0]:
                raise ValueError(f"{name} must contain (min, max) with max > min")
            if not all(math.isfinite(float(value)) for value in axis_range):
                raise ValueError(f"{name} values must be finite")
        if self.height < 1 or self.width < 1:
            raise ValueError("BEV height and width must be positive")


def filter_points_to_grid(
    points: torch.Tensor,
    grid: PointGridSpec,
) -> torch.Tensor:
    """Filter `(N, >=3)` points to a PointGridSpec's half-open XYZ bounds."""
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Points must have shape (N, >=3), got {tuple(points.shape)}")
    xyz = points[:, :3]
    valid = (
        (xyz[:, 0] >= grid.min_xyz[0])
        & (xyz[:, 0] < grid.max_xyz[0])
        & (xyz[:, 1] >= grid.min_xyz[1])
        & (xyz[:, 1] < grid.max_xyz[1])
        & (xyz[:, 2] >= grid.min_xyz[2])
        & (xyz[:, 2] < grid.max_xyz[2])
    )
    return points[valid]


def points_to_voxels(
    points: torch.Tensor,
    batch: torch.Tensor,
    grid: PointGridSpec,
    batch_size: int,
    feature_dim: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Aggregate per-batch XYZ/feature points into unique grid voxels.

    Returns PTv3's `coord`, `feat`, `grid_coord`, `batch` data contract plus a
    real-point mask. Empty batch entries receive a dummy voxel marked unreal.
    """
    if points.ndim != 2 or points.shape[1] < feature_dim:
        raise ValueError(f"Points must have shape (N, >={feature_dim}), got {tuple(points.shape)}")
    if batch.ndim != 1 or batch.shape[0] != points.shape[0]:
        raise ValueError("batch must contain one scene ID per point")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if torch.any((batch < 0) | (batch >= batch_size)):
        raise ValueError("batch IDs must be within [0, batch_size)")

    xyz_all = points[:, :3]
    keep = (
        (xyz_all[:, 0] >= grid.min_xyz[0])
        & (xyz_all[:, 0] < grid.max_xyz[0])
        & (xyz_all[:, 1] >= grid.min_xyz[1])
        & (xyz_all[:, 1] < grid.max_xyz[1])
        & (xyz_all[:, 2] >= grid.min_xyz[2])
        & (xyz_all[:, 2] < grid.max_xyz[2])
    )
    points = points[keep]
    batch = batch[keep]
    raw_features = points[:, :feature_dim].to(dtype=torch.float32)
    nx, ny, nz = grid.dimensions
    dimensions = raw_features.new_tensor(grid.voxel_size)
    xyz = raw_features[:, :3]
    grid_coord = torch.stack(
        (
            torch.clamp(((xyz[:, 0] - grid.min_xyz[0]) / dimensions[0]).floor().long(), 0, nx - 1),
            torch.clamp(((xyz[:, 1] - grid.min_xyz[1]) / dimensions[1]).floor().long(), 0, ny - 1),
            torch.clamp(((xyz[:, 2] - grid.min_xyz[2]) / dimensions[2]).floor().long(), 0, nz - 1),
        ),
        dim=-1,
    )

    if raw_features.shape[0]:
        keys = (((batch * nx + grid_coord[:, 0]) * ny + grid_coord[:, 1]) * nz) + grid_coord[:, 2]
        _, inverse, counts = torch.unique(keys, sorted=True, return_inverse=True, return_counts=True)
        voxel_feat = raw_features.new_zeros(counts.numel(), feature_dim)
        voxel_feat.index_add_(0, inverse, raw_features)
        voxel_feat /= counts.unsqueeze(-1).to(voxel_feat.dtype)
        voxel_coord = voxel_feat[:, :3]
        unique_keys, inverse, counts = torch.unique(
            keys,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        voxel_grid_z = unique_keys % nz
        tmp = unique_keys // nz

        voxel_grid_y = tmp % ny
        tmp = tmp // ny

        voxel_grid_x = tmp % nx
        voxel_batch = tmp // nx

        voxel_grid = torch.stack(
            [voxel_grid_x, voxel_grid_y, voxel_grid_z],
            dim=-1,
        )
        is_real = torch.ones(counts.numel(), dtype=torch.bool, device=raw_features.device)
    else:
        voxel_coord = raw_features.new_empty((0, 3))
        voxel_feat = raw_features.new_empty((0, feature_dim))
        voxel_batch = torch.empty(0, dtype=torch.long, device=raw_features.device)
        voxel_grid = torch.empty(0, 3, dtype=torch.long, device=raw_features.device)
        is_real = torch.empty(0, dtype=torch.bool, device=raw_features.device)

    present = torch.bincount(voxel_batch, minlength=batch_size)
    empty_batches = torch.where(present == 0)[0]
    if empty_batches.numel():
        dummy_xyz = raw_features.new_tensor(
            tuple(minimum + voxel_size / 2 for minimum, voxel_size in zip(grid.min_xyz, grid.voxel_size, strict=True))
        ).expand(empty_batches.numel(), -1)
        voxel_coord = torch.cat([voxel_coord, dummy_xyz])
        dummy_feat = dummy_xyz.new_zeros(empty_batches.numel(), feature_dim)
        dummy_feat[:, :3] = dummy_xyz
        voxel_feat = torch.cat([voxel_feat, dummy_feat])
        voxel_batch = torch.cat([voxel_batch, empty_batches])
        voxel_grid = torch.cat(
            [voxel_grid, torch.zeros(empty_batches.numel(), 3, dtype=torch.long, device=raw_features.device)]
        )
        is_real = torch.cat([is_real, torch.zeros(empty_batches.numel(), dtype=torch.bool, device=raw_features.device)])

    return {
        "coord": voxel_coord,
        "feat": voxel_feat,
        "grid_coord": voxel_grid,
        "batch": voxel_batch,
    }, is_real


def scatter_ptv3_features_to_bev(
    features: torch.Tensor,
    batch: torch.Tensor,
    xyz: torch.Tensor,
    bev: BEVGridSpec,
    batch_size: int,
) -> torch.Tensor:
    """Scatter point features to XY BEV using one cell per batch item."""
    if features.shape[0] != batch.shape[0] or features.shape[0] != xyz.shape[0]:
        raise ValueError("features, batch, and xyz must have the same leading dimension")
    if xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    ix = (((xyz[:, 0] - bev.x_range[0]) / (bev.x_range[1] - bev.x_range[0])) * bev.width).long()
    iy = (((xyz[:, 1] - bev.y_range[0]) / (bev.y_range[1] - bev.y_range[0])) * bev.height).long()
    valid = (ix >= 0) & (ix < bev.width) & (iy >= 0) & (iy < bev.height)
    flat_index = batch[valid] * (bev.height * bev.width) + iy[valid] * bev.width + ix[valid]
    num_cells = batch_size * bev.height * bev.width
    bev_flat = features.new_zeros(num_cells, features.shape[-1])
    bev_flat.index_add_(0, flat_index, features[valid])
    counts = torch.bincount(flat_index, minlength=num_cells).to(features.dtype).unsqueeze(-1)
    return bev_flat / counts.clamp_min(1)


def adapt_ptv3_config_for_input(
    config: PointTransformerV3Config,
    grid: PointGridSpec,
) -> PointTransformerV3Config:
    """Adapt serialization capacity and validate stage settings for a point grid."""
    depth = grid.serial_depth
    if depth > 21:
        raise ValueError(
            f"Point grid dimensions {grid.dimensions} require serial_depth={depth}, "
            "but PTv3 supports at most serial_depth=21"
        )
    return replace(config, serial_depth=depth)


def adapt_ptv3_stride_for_output(
    config: PointTransformerV3Config,
    input_grid: PointGridSpec,
    output_voxel_size_xyz: tuple[float, float, float] | float,
) -> tuple[int, ...]:
    """Return stage strides that approximate a physical isotropic output voxel size.

    PTv3 currently divides X, Y, and Z grid coordinates by the same integer stride,
    so output voxel sizes must scale uniformly. For anisotropic output grids, add
    per-axis pooling support before using this helper.
    """
    if isinstance(output_voxel_size_xyz, (int, float)):
        output_voxel_size = (
            float(output_voxel_size_xyz),
            float(output_voxel_size_xyz),
            float(output_voxel_size_xyz),
        )
    else:
        if len(output_voxel_size_xyz) != 3:
            raise ValueError("output_voxel_size_xyz must be a scalar or contain exactly three values")
        output_voxel_size = tuple(float(v) for v in output_voxel_size_xyz)
    if not all(math.isfinite(v) and v > 0 for v in output_voxel_size):
        raise ValueError("Output voxel size values must be positive and finite")

    ratios = [
        output_size / input_size
        for output_size, input_size in zip(output_voxel_size, input_grid.voxel_size, strict=True)
    ]
    if max(ratios) / min(ratios) > 1.001:
        raise ValueError(f"PTv3 scalar stride requires isotropic output scaling; got input/output ratios {ratios}")

    total_stride = max(1, round(ratios[0]))
    if not math.isclose(ratios[0], total_stride, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError(
            f"Output voxel ratio {ratios[0]:.6g} must be an integer because GridPooling uses integer strides"
        )

    transitions = len(config.enc_depths) - 1
    if transitions < 1:
        if total_stride != 1:
            raise ValueError("A one-stage encoder cannot downsample the output grid")
        return ()

    target = total_stride ** (1.0 / transitions)
    best: tuple[int, ...] | None = None
    best_score = float("inf")

    def allocate(position: int, remaining: int, chosen: tuple[int, ...]) -> None:
        nonlocal best, best_score
        slots_left = transitions - position
        if slots_left == 0:
            if remaining == 1:
                score = sum((math.log(stride) - math.log(target)) ** 2 for stride in chosen)
                if score < best_score:
                    best_score = score
                    best = chosen
            return
        if slots_left == 1:
            if remaining >= 1:
                allocate(position + 1, 1, (*chosen, remaining))
            return
        divisors: list[int] = []
        limit = math.isqrt(remaining)
        for stride in range(1, limit + 1):
            if remaining % stride == 0:
                divisors.append(stride)
                complement = remaining // stride
                if complement != stride:
                    divisors.append(complement)
        for stride in sorted(set(divisors)):
            allocate(position + 1, remaining // stride, (*chosen, stride))

    allocate(0, total_stride, ())
    if best is None:
        raise ValueError(f"Output stride factor {total_stride} cannot be allocated across {transitions} PTv3 stages")
    return best


def adapt_ptv3_config_for_output(
    config: PointTransformerV3Config,
    input_grid: PointGridSpec,
    output_voxel_size_xyz: tuple[float, float, float] | float,
) -> PointTransformerV3Config:
    """Set PTv3 stage strides for the requested physical output voxel size."""
    stride = adapt_ptv3_stride_for_output(config, input_grid, output_voxel_size_xyz)
    return replace(config, stride=stride)


# --------------------------------------------------------------------------- #
# Point container
# --------------------------------------------------------------------------- #
class Point(dict):
    """dict with attribute access: point.feat, point.coord, point.batch, ...

    A thin container that is passed through the whole network. Layers read and write
    entries by attribute (``point.feat = ...``) which keeps the call signatures short.
    Missing keys raise ``AttributeError`` (not ``KeyError``) on attribute access so that
    ``hasattr`` / ``getattr(..., default)`` behave as expected; dunder lookups are
    rejected so copy / pickle protocols do not accidentally hit the dict.
    """

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _spread3(x: torch.Tensor) -> torch.Tensor:
    """Spread 21 coordinate bits into positions separated by two zero positions.

    Bit ``i`` of ``x`` moves to bit ``3 * i`` of the result (``... b2 b1 b0`` becomes
    ``... 00b2 00b1 00b0``), leaving room to interleave two further coordinates. Only the
    low 21 bits are used, so the result fits into a non-negative int64 (63 bits). Written
    with ``//`` and ``remainder`` instead of bit shifts so it runs on any backend.
    """
    x = x.remainder(1 << 21)
    out = torch.zeros_like(x)
    for i in range(21):
        out = out + ((x // (1 << i)).remainder(2) * (1 << (3 * i)))
    return out


def _interleave(x0, x1, x2):
    """Morton-interleave three integer coordinates (``x0`` is the most significant axis).

    The result has bit layout ``... x0_i x1_i x2_i ... x0_0 x1_0 x2_0``.
    """
    return (_spread3(x0) * 4) + (_spread3(x1) * 2) + _spread3(x2)


def z_order_encode(gc: torch.Tensor, depth: int) -> torch.Tensor:
    """Z-order (Morton) code of integer grid coordinates.

    Args:
        gc: ``(N, 3)`` integer voxel coordinates.
        depth: Bits per axis. Unused here (the interleave always spreads 21 bits) but kept
            so that all encoders share the same signature.

    Returns:
        ``(N,)`` int64 codes; sorting by them visits the voxels along a Z-shaped curve.
    """
    gc = gc.long()
    return _interleave(gc[:, 0], gc[:, 1], gc[:, 2])


def hilbert_encode(gc: torch.Tensor, depth: int) -> torch.Tensor:
    """3D Hilbert index via Skilling's transpose algorithm (vectorized, out-of-place).

    Compared with Z-order, the Hilbert curve never jumps: consecutive indices are always
    spatially adjacent voxels, which gives better locality inside a patch.

    Args:
        gc: ``(N, 3)`` integer voxel coordinates in ``[0, 2**depth)``.
        depth: Bits per axis.

    Returns:
        ``(N,)`` int64 Hilbert indices.

    The algorithm works on the "transposed" Hilbert index in three steps:
    (1) undo excess work, i.e. invert / exchange low bits for every bit plane from the
    most significant one down, (2) Gray-encode the coordinates, (3) fix up the lowest bits
    with a parity mask ``t``. Finally the transposed form is bit-interleaved into a single
    integer. All per-point branches are expressed with ``torch.where`` so the code is
    fully vectorized and never mutates its input.
    """
    X = [gc[:, i].long() for i in range(3)]
    M = 1 << (depth - 1)
    # inverse undo excess work
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(3):
            has = (X[i] & Q) != 0
            if i == 0:
                X[0] = torch.where(has, X[0] ^ P, X[0])  # invert
            else:
                x0_inv = torch.where(has, X[0] ^ P, X[0])  # invert
                t = torch.where(has, torch.zeros_like(X[0]), (X[0] ^ X[i]) & P)  # exchange
                X[0] = x0_inv ^ t
                X[i] = X[i] ^ t
        Q >>= 1
    # Gray encode
    X[1] = X[1] ^ X[0]
    X[2] = X[2] ^ X[1]
    t = torch.zeros_like(X[0])
    Q = M
    while Q > 1:
        t = torch.where((X[2] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    X = [x ^ t for x in X]
    return _interleave(*X)


@torch.no_grad()
def serialize(point: Point, order: Sequence[str], depth: int, shuffle_orders: bool) -> Point:
    """Compute serialization orders + per-stage patch bookkeeping (cloud sizes, caches).

    Every point gets one integer code per requested order. The batch index is placed in
    the high bits, so sorting the codes keeps each cloud contiguous and ordered by cloud
    id, while the low bits order the points along the space-filling curve inside a cloud.

    Args:
        point: Must contain ``grid_coord`` (N, 3) and ``batch`` (N,).
        order: Names of the orders to compute, e.g. ``("z", "hilbert-trans")``. A
            ``-trans`` suffix reverses the axis order (x<->z) before encoding, which yields
            a differently oriented curve from the same base pattern.
        depth: Bits per axis; coordinates must be in ``[0, 2**depth)`` and
            ``depth <= 21``.
        shuffle_orders: Randomly permute the rows of the code matrix (i.e. the assignment
            of orders to blocks). Pass False at inference for deterministic behaviour.

    Returns:
        The same ``point`` with these entries added:

        * ``serial_order``   ``(n_orders, N)``: ``argsort`` of the codes; entry ``[o, j]``
          is the index of the point at position ``j`` of order ``o``.
        * ``serial_inverse`` ``(n_orders, N)``: inverse permutation, i.e. the position of
          each point in order ``o``.
        * ``cloud_sizes``    ``(B,)``: points per cloud (ascending batch id).
        * ``min_cloud``      smallest cloud size; upper bound for the patch size.
        * ``layout_cache``   empty dict, filled lazily by ``patch_layout``.
    """
    assert depth <= 21, "depth > 21 overflows int64 Morton codes"
    gc, batch = point.grid_coord.long(), point.batch.long()
    assert int(gc.min()) >= 0 and int(gc.max()) < (1 << depth), "grid_coord out of range for depth"
    assert int(batch.max()).bit_length() + 3 * depth <= 62, "batch index + code does not fit in int64"
    codes = []
    for o in order:
        g = gc[:, [2, 1, 0]] if o.endswith("-trans") else gc
        base = o.replace("-trans", "")
        if base == "z":
            c = z_order_encode(g, depth)
        elif base == "hilbert":
            c = hilbert_encode(g, depth)
        else:
            raise ValueError(f"unknown order {o}")
        codes.append((batch * (1 << (3 * depth))) + c)  # batch in the high bits keeps clouds contiguous
    codes = torch.stack(codes)
    if shuffle_orders:
        codes = codes[torch.randperm(codes.shape[0], device=codes.device)]
    perm = torch.argsort(codes, dim=1)
    inverse = torch.empty_like(perm).scatter_(1, perm, torch.arange(perm.shape[1], device=perm.device).expand_as(perm))
    point.serial_order, point.serial_inverse = perm, inverse
    # per-stage patch bookkeeping; clouds appear in ascending batch id in the sorted order
    _, sizes = torch.unique(batch, sorted=True, return_counts=True)
    point.cloud_sizes = sizes
    point.min_cloud = int(sizes.min())
    point.layout_cache = {}
    return point


# --------------------------------------------------------------------------- #
# Submanifold sparse convolution via neighbour lookup
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_neighbors(
    grid_coord: torch.Tensor,
    batch: torch.Tensor,
    kernel_size: int = 3,
    grid_bound: int | None = None,
) -> torch.Tensor:
    """(N, k^3) indices of the neighbours among occupied voxels; N marks 'missing'.

    For every voxel, look up the ``k³`` voxels of its ``k x k x k`` neighbourhood (same
    cloud only) among the occupied ones. This is the "rulebook" of a submanifold sparse
    convolution: the output sites equal the input sites, so only existing voxels are
    gathered.

    Lookup strategy: every voxel is packed into a single int64 key
    ``((batch * S + x) * S + y) * S + z``; the keys are sorted once and each offset's
    query keys are located with ``searchsorted`` (a vectorized binary search). Offsets are
    processed in chunks of 16 to bound peak memory (``16 * N`` queries at a time).

    Args:
        grid_coord: ``(N, 3)`` integer voxel coordinates (must be >= 0).
        batch: ``(N,)`` cloud id of every voxel.
        kernel_size: Odd kernel size ``k``.
        grid_bound: Side length ``S`` of the hash grid. Must exceed every coordinate
            (plus padding). Defaults to ``max(grid_coord) + k``; passing a fixed bound
            (e.g. ``2**serial_depth``) makes the keys independent of the data.

    Returns:
        ``(N, k³)`` long tensor. Column ``j`` corresponds to offset ``j`` of the
        row-major ``(dx, dy, dz)`` enumeration over ``[-p, p]³`` (``p = k // 2``), the
        same ordering that ``SubMConv3d.weight`` uses. Missing neighbours (empty voxel or
        outside the grid) hold the sentinel ``N``.

    Under ONNX export ``searchsorted`` is replaced by an explicit 64-step binary search,
    since the operator is not exported reliably.
    """
    assert kernel_size % 2 == 1
    N = grid_coord.shape[0]
    p = kernel_size // 2
    gc = grid_coord.long()
    b = batch.long()
    S = grid_bound if grid_bound is not None else int(gc.max()) + 2 * p + 1

    def _key(bb, c):
        return ((bb * S + c[..., 0]) * S + c[..., 1]) * S + c[..., 2]

    sk, perm = _key(b, gc).sort()
    sk = torch.cat((sk, sk.new_full((1,), torch.iinfo(sk.dtype).max)))  # sentinel so pos == N stays in bounds
    perm = torch.cat((perm, perm.new_full((1,), N)))
    r = torch.arange(-p, p + 1, device=gc.device)
    offs = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    out = torch.empty((N, offs.shape[0]), dtype=torch.long, device=grid_coord.device)
    for start in range(0, offs.shape[0], 16):
        offset_chunk = offs[start : start + 16]
        query_coord = gc[None, :, :] + offset_chunk[:, None, :]
        valid = ((query_coord >= 0) & (query_coord < S)).all(dim=-1)  # neighbours outside the grid are missing
        q = _key(b[None, :], query_coord.clamp(0, S - 1))
        q_flat = q.reshape(-1)
        if torch.onnx.is_in_onnx_export():
            # manual lower-bound binary search (equivalent to searchsorted, left side)
            lo = torch.zeros_like(q_flat)
            hi = torch.full_like(q_flat, N)
            for _ in range(64):
                mid = (lo + hi) // 2
                mid_key = sk[mid]
                lo = torch.where(mid_key < q_flat, mid + 1, lo)
                hi = torch.where(mid_key < q_flat, hi, mid)
            pos = lo.reshape_as(q)
        else:
            pos = torch.searchsorted(sk, q_flat).reshape_as(q)
        found = (sk[pos] == q) & valid  # key must match exactly, otherwise the voxel is empty
        out[:, start : start + offset_chunk.shape[0]] = torch.where(found, perm[pos], torch.full_like(pos, N)).T
    return out


class SubMConv3d(nn.Module):
    """Submanifold sparse conv (stride 1) == spconv.SubMConv3d, via gather + matmul.

    Output sites are identical to input sites, so the sparsity pattern never dilates.
    Instead of a dense 3D grid, features are gathered through a precomputed neighbour
    table (see ``build_neighbors``) and multiplied by one ``(in_dim, out_dim)`` weight
    matrix per kernel offset:

        out[i] = sum_j  feat[nbr[i, j]] @ W[j]  (+ bias)

    with missing neighbours contributing zero.

    Args:
        in_dim: Input channels.
        out_dim: Output channels.
        kernel_size: Odd kernel size ``k``; the layer has ``k³`` offsets.
        bias: Whether to add a learnable bias.

    Weights are initialised with He-style normal noise using ``k³ * in_dim`` as fan-in.
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.k3 = kernel_size**3
        self.weight = nn.Parameter(torch.empty(self.k3, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.normal_(self.weight, std=(2.0 / (self.k3 * in_dim)) ** 0.5)

    def forward(self, feat: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        """Apply the sparse convolution.

        Args:
            feat: ``(N, in_dim)`` voxel features.
            nbr: ``(N, k³)`` neighbour table from ``build_neighbors`` (sentinel ``N``).

        Returns:
            ``(N, out_dim)`` features. Offsets are processed 8 at a time so that the
            gathered tensor ``(N, 8, in_dim)`` stays small instead of ``(N, k³, in_dim)``.
        """
        padded = torch.cat([feat, feat.new_zeros(1, feat.shape[1])], dim=0)  # row N = zeros for missing neighbours
        out = feat.new_zeros(feat.shape[0], self.out_dim, dtype=feat.dtype)
        for start in range(0, self.k3, 8):
            stop = min(start + 8, self.k3)
            gathered = padded[nbr[:, start:stop]].permute(1, 0, 2)  # (chunk, N, in_dim)
            out = out + torch.bmm(gathered, self.weight[start:stop]).sum(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class DropPath(nn.Module):
    """Stochastic depth, per point (as in the official code, which applies timm's DropPath to (N, C)).

    During training the whole residual branch is zeroed for a random subset of points
    (probability ``p``) and surviving points are rescaled by ``1 / (1 - p)`` so the
    expectation is unchanged. Identity in eval mode.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = x.new_empty(x.shape[0], 1).bernoulli_(keep) / keep
        return x * mask


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that also accepts a single row in training mode.

    A sparse batch can shrink to one voxel after grid pooling, for which batch statistics
    are undefined. The running statistics are used (and left unchanged) in that case.
    Parameter and buffer names equal ``nn.BatchNorm1d``, so checkpoints stay compatible.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.shape[0] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


class MLP(nn.Module):
    """Two-layer feed-forward network ``Linear -> GELU -> Linear`` with dropout.

    Args:
        dim: Input and output width.
        hidden: Hidden width (typically ``mlp_ratio * dim``).
        drop: Dropout probability applied after the activation and after the output layer.
    """

    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


def _pad_unpad(bc: torch.Tensor, ps: int):
    """bc: (B,) cloud sizes, all >= ps. Pad every cloud to a multiple of ps by re-using the previous patch.

    Attention works on equally sized patches, so each cloud's length is rounded up to a
    multiple of ``ps``. The missing slots are filled by re-using points of the preceding
    patch of the same cloud (instead of zeros), so no mask is needed. Example with a cloud
    of 5 points and ``ps = 4``: the padded cloud has 8 slots holding points
    ``[0 1 2 3 | 4 1 2 3]``; the last patch therefore contains 4 valid-looking points.

    Returns:
        pad:   ``(N_pad,)`` index into the *serialized* (sorted) point sequence for every
            padded slot.
        unpad: ``(N,)`` position in the padded sequence of every sorted point (the
            first/real occurrence), used to scatter results back.

    Requires ``bc >= ps`` for every cloud (otherwise there is no previous patch to copy);
    ``patch_layout`` guarantees this by clipping ``ps`` to the smallest cloud.
    """
    dev, B = bc.device, bc.numel()
    bc_pad = (bc + ps - 1) // ps * ps
    off = F.pad(bc.cumsum(0), (1, 0))
    off_pad = F.pad(bc_pad.cumsum(0), (1, 0))
    ids = torch.arange(B, device=dev)
    cid_pad = torch.repeat_interleave(ids, bc_pad)
    local = torch.arange(int(off_pad[-1]), device=dev) - off_pad[cid_pad]
    local = torch.where(local >= bc[cid_pad], local - ps, local)  # reuse previous patch
    pad = off[cid_pad] + local
    cid = torch.repeat_interleave(ids, bc)
    unpad = torch.arange(int(off[-1]), device=dev) - off[cid] + off_pad[cid]
    return pad, unpad


@torch.no_grad()
def patch_layout(point: Point, max_patch: int, order_index: int):
    """Cached (patch size, gather order into padded patches, scatter-back index).

    Args:
        point: Point with the bookkeeping from ``serialize``.
        max_patch: Requested patch size; clipped to ``point.min_cloud``.
        order_index: Which serialization order (row of ``serial_order``) to use.

    Returns:
        ps:      effective patch size.
        order:   ``(N_pad,)`` original point indices laid out in padded, serialized order;
            ``feat[order]`` can be reshaped to ``(num_patches, ps, C)``.
        inverse: ``(N,)`` for every original point, its row in the padded layout; gathers
            the attention output back to the original point order.

    Results are memoised in ``point.layout_cache`` (keyed by order index and patch size),
    so blocks of the same stage that share an order and patch size reuse the layout. The
    cache lives on the ``Point`` of a stage and is reset by ``serialize``.
    """
    ps = min(max_patch, point.min_cloud)  # a patch can't exceed the smallest cloud
    cache = point.layout_cache
    key = (order_index, ps)
    if key not in cache:
        if ("pad", ps) not in cache:
            cache[("pad", ps)] = _pad_unpad(point.cloud_sizes, ps)
        pad, unpad = cache[("pad", ps)]
        cache[key] = (
            point.serial_order[order_index][pad],
            unpad[point.serial_inverse[order_index]],
        )
    order, inverse = cache[key]
    return ps, order, inverse


class SerializedAttention(nn.Module):
    """Multi-head self-attention within patches of the serialized point sequence.

    Points are sorted along a space-filling curve, cut into consecutive patches of
    ``patch_size`` points, and full attention is computed inside each patch only. Because
    the curve preserves locality, a patch is a compact spatial region, and the cost is
    linear in ``N``. There is no positional encoding here; position information comes
    from the xCPE conv in ``Block``. Padding duplicates (see ``_pad_unpad``) take part in
    attention as ordinary points; no mask is applied.

    Args:
        dim: Channel width (must be divisible by ``heads``).
        heads: Number of attention heads.
        patch_size: Maximum patch size (clipped to the smallest cloud at run time).
        order_index: Row of ``point.serial_order`` this layer uses; blocks of a stage cycle
            through the orders so that patch borders differ from block to block.
        qk_norm: LayerNorm over the head dimension of q and k (optional extension).
        attn_drop: Dropout on attention probabilities (training only).
        proj_drop: Dropout after the output projection.
    """

    def __init__(self, dim, heads, patch_size, order_index, qk_norm=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.patch_size, self.order_index = dim, heads, patch_size, order_index
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        hd = dim // heads
        self.q_norm = nn.LayerNorm(hd) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(hd) if qk_norm else nn.Identity()
        self.attn_drop, self.proj_drop = attn_drop, nn.Dropout(proj_drop)

    def forward(self, feat: torch.Tensor, point: Point) -> torch.Tensor:
        """Args: ``feat`` (N, C) features in original point order; ``point`` with
        serialization bookkeeping. Returns ``(N, C)`` attention output in original order.

        Shape walk-through: ``(N, 3C)`` qkv -> gathered to padded serial order
        ``(N_pad, 3C)`` -> ``(P, ps, 3, H, d)`` -> ``(3, P, H, ps, d)`` with ``P`` patches,
        ``H`` heads, ``d = C / H``. SDPA then runs batched over ``(P, H)``.
        """
        C = feat.shape[1]
        ps, order, inverse = patch_layout(point, self.patch_size, self.order_index)
        qkv = self.qkv(feat)[order]  # (N_pad, 3C)
        qkv = qkv.reshape(-1, ps, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (P, H, ps, d)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(-1, C)[inverse]  # back to (N, C) in original point order
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    """One PTv3 transformer block (pre-norm, three residual branches).

    ``x = x + LN(Linear(SubMConv3x3x3(x)))``   xCPE: conditional positional encoding
    ``x = x + DropPath(Attn(LN(x)))``          serialized patch attention
    ``x = x + DropPath(MLP(LN(x)))``           feed-forward

    The block reads ``point.feat`` and ``point.nbr`` (neighbour table of the current
    resolution) plus the serialization bookkeeping, and updates ``point.feat``.

    Args:
        dim: Channel width.
        heads: Attention heads.
        patch_size: Maximum attention patch size.
        order_index: Serialization order used by this block's attention.
        mlp_ratio: MLP hidden width relative to ``dim``.
        qk_norm: Enable q/k LayerNorm in attention.
        attn_drop: Attention dropout.
        proj_drop: Projection / MLP dropout.
        drop_path: Stochastic depth rate of this specific block.
    """

    def __init__(
        self,
        dim,
        heads,
        patch_size,
        order_index,
        mlp_ratio=4.0,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.cpe_conv = SubMConv3d(dim, dim, 3, bias=True)
        self.cpe_lin = nn.Linear(dim, dim)
        self.cpe_norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SerializedAttention(dim, heads, patch_size, order_index, qk_norm, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), proj_drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, point: Point) -> Point:
        point.feat = point.feat + self.cpe_norm(self.cpe_lin(self.cpe_conv(point.feat, point.nbr)))
        point.feat = point.feat + self.drop_path(self.attn(self.norm1(point.feat), point))
        point.feat = point.feat + self.drop_path(self.mlp(self.norm2(point.feat)))
        return point


class GridPooling(nn.Module):
    """Voxel pooling: Linear -> max over cells -> BN -> GELU; coords are averaged.

    Merges all points that fall into the same coarse voxel (``grid_coord // stride``, per
    cloud) into one point of the next, coarser stage. Features are projected first and
    then max-pooled per cell; continuous coordinates are averaged.

    Args:
        in_dim: Input channels.
        out_dim: Output channels (width of the next stage).
        stride: Integer downsampling factor of the voxel grid (>= 1).

    Returns (from ``forward``): a new ``Point`` with ``feat`` (M, out_dim), ``coord``
    (M, 3), ``grid_coord`` (M, 3) at the coarse resolution, ``batch`` (M,), plus
    ``pooling_parent`` (the finer ``Point``) and ``pooling_inverse`` (N,) mapping each
    fine point to its coarse cell; both are consumed by ``GridUnpooling``. The new points
    are ordered by their packed voxel key (cloud, then x, y, z).
    """

    def __init__(self, in_dim, out_dim, stride=2):
        super().__init__()
        assert stride >= 1
        self.stride = stride
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = SafeBatchNorm1d(out_dim)

    def forward(self, point: Point) -> Point:
        gc = point.grid_coord.long() // self.stride
        b = point.batch.long()
        S = int(gc.max()) + 1
        key = ((b * S + gc[:, 0]) * S + gc[:, 1]) * S + gc[:, 2]  # unique id per (cloud, coarse voxel)
        uniq, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
        M = uniq.shape[0]

        feat = self.proj(point.feat)
        # every cell has at least one point, so the -inf initial value is always overwritten
        pooled = torch.full((M, feat.shape[1]), float("-inf"), device=feat.device, dtype=feat.dtype).scatter_reduce(
            0, inv[:, None].expand_as(feat), feat, reduce="amax", include_self=True
        )
        coord = torch.zeros(M, 3, device=feat.device, dtype=point.coord.dtype).index_add_(0, inv, point.coord)
        coord = coord / counts[:, None].to(coord.dtype)

        # Decode the unique voxel keys instead of reducing int64 point indices.
        # Metal does not support the int64 atomic operation used by scatter_reduce_
        # for selecting a representative point.
        new_b = uniq // (S**3)
        new_gc = torch.stack(
            ((uniq // (S**2)) % S, (uniq // S) % S, uniq % S),
            dim=1,
        )

        return Point(
            feat=F.gelu(self.norm(pooled)),
            coord=coord,
            grid_coord=new_gc,
            batch=new_b,
            pooling_parent=point,
            pooling_inverse=inv,
        )


class GridUnpooling(nn.Module):
    """Decoder upsampling: parent.feat = proj_skip(parent.feat) + proj(child.feat)[inverse].

    Returns to the finer resolution recorded by ``GridPooling``. The coarse features are
    projected and broadcast to every fine point of their cell (via ``pooling_inverse``),
    then added to a projection of the fine point's own (skip-connection) features. Each
    projection is ``Linear -> BatchNorm -> GELU``.

    Args:
        in_dim: Channels of the coarse (child) features.
        skip_dim: Channels of the fine (parent) features, i.e. the encoder stage width.
        out_dim: Output channels (width of the decoder stage).

    Note that the parent ``Point`` is modified in place and returned: its ``feat`` is
    replaced and its other entries (coordinates, serialization, neighbour table, link to
    its own parent) are reused for the decoder blocks of that stage.
    """

    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), SafeBatchNorm1d(out_dim), nn.GELU())
        self.proj_skip = nn.Sequential(nn.Linear(skip_dim, out_dim), SafeBatchNorm1d(out_dim), nn.GELU())

    def forward(self, point: Point) -> Point:
        parent, inverse = point.pooling_parent, point.pooling_inverse
        parent.feat = self.proj_skip(parent.feat) + self.proj(point.feat)[inverse]
        return parent


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class PointTransformerV3Encoder(nn.Module):
    """Point Transformer V3 backbone: sparse stem, hierarchical encoder, optional decoder.

    Despite its name the class also builds the U-Net decoder unless ``enc_mode=True``.
    See ``PointTransformerV3Config`` for all hyper-parameters; the defaults reproduce the
    canonical PTv3 layout (5 stages, depths 2-2-2-6-2, widths 32...512).

    Attributes set on the instance:
        out_channels: Width of ``output.feat`` (``dec_channels[0]``, or the last encoder
            width when ``enc_mode=True``).
        pools / enc: Grid poolings and encoder block lists (one list per stage).
        dec_up / dec: Unpoolings and decoder block lists, ordered from the deepest stage
            to the shallowest, i.e. in execution order.

    Example:
        >>> model = PointTransformerV3Encoder(PointTransformerV3Config(in_channels=6))
        >>> out = model(dict(coord=coord, feat=feat, grid_coord=gc, batch=batch))
        >>> out.feat.shape  # (N, 64)
    """

    def __init__(self, config: PointTransformerV3Config | None = None):
        super().__init__()
        config = config or PointTransformerV3Config()
        n = len(config.enc_depths)
        stride = tuple(config.stride) if config.stride is not None else (2,) * (n - 1)
        config = replace(config, stride=stride)
        self.config = config
        in_channels = config.in_channels
        order = config.order
        enc_depths = config.enc_depths
        enc_channels = config.enc_channels
        enc_num_head = config.enc_num_head
        enc_patch_size = config.enc_patch_size
        dec_depths = config.dec_depths
        dec_channels = config.dec_channels
        dec_num_head = config.dec_num_head
        dec_patch_size = config.dec_patch_size
        mlp_ratio = config.mlp_ratio
        qk_norm = config.qk_norm
        attn_drop = config.attn_drop
        proj_drop = config.proj_drop
        drop_path = config.drop_path
        shuffle_orders = config.shuffle_orders
        serial_depth = config.serial_depth
        stem_kernel = config.stem_kernel
        enc_mode = config.enc_mode

        if stem_kernel <= 0 or stem_kernel % 2 == 0:
            raise ValueError("stem_kernel must be a positive odd integer")
        if len(stride) != n - 1 or len(enc_channels) != n or len(enc_num_head) != n or len(enc_patch_size) != n:
            raise ValueError("stride must have one value per stage transition; encoder settings must match stages")
        self.in_channels = in_channels
        self.order, self.shuffle_orders, self.serial_depth = list(order), shuffle_orders, serial_depth
        self.num_stages, self.enc_mode, self.stem_kernel = n, enc_mode, stem_kernel
        n_ord = len(self.order)

        # stem: sparse 5x5x5 conv + BN + GELU
        self.embed_conv = SubMConv3d(in_channels, enc_channels[0], stem_kernel, bias=False)
        self.embed_norm = SafeBatchNorm1d(enc_channels[0])

        # encoder: stage s = [GridPooling from s-1 (if s > 0)] + enc_depths[s] blocks.
        # Stochastic-depth rates rise linearly over all encoder blocks; block i of a stage
        # uses serialization order i % n_ord.
        dpr = torch.linspace(0, drop_path, sum(enc_depths)).tolist()
        self.pools = nn.ModuleList()
        self.enc = nn.ModuleList()
        c = 0
        for s in range(n):
            if s > 0:
                self.pools.append(GridPooling(enc_channels[s - 1], enc_channels[s], stride[s - 1]))
            self.enc.append(
                nn.ModuleList(
                    Block(
                        enc_channels[s],
                        enc_num_head[s],
                        enc_patch_size[s],
                        i % n_ord,
                        mlp_ratio,
                        qk_norm,
                        attn_drop,
                        proj_drop,
                        dpr[c + i],
                    )
                    for i in range(enc_depths[s])
                )
            )
            c += enc_depths[s]
        self.out_channels = enc_channels[-1]

        # decoder
        if not enc_mode:
            assert len(dec_depths) == len(dec_channels) == len(dec_num_head) == len(dec_patch_size) == n - 1
            # dec_ch[s] = width at decoder stage s; the deepest "stage" is the encoder output
            dec_ch = list(dec_channels) + [enc_channels[-1]]
            ddpr = torch.linspace(0, drop_path, sum(dec_depths)).tolist()
            self.dec_up = nn.ModuleList()  # built from the deepest stage to the shallowest
            self.dec = nn.ModuleList()
            for s in reversed(range(n - 1)):
                self.dec_up.append(GridUnpooling(dec_ch[s + 1], enc_channels[s], dec_ch[s]))
                # drop-path rates of this stage's slice, reversed so the deepest blocks get the highest rate
                dp = ddpr[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])][::-1]
                self.dec.append(
                    nn.ModuleList(
                        Block(
                            dec_ch[s],
                            dec_num_head[s],
                            dec_patch_size[s],
                            i % n_ord,
                            mlp_ratio,
                            qk_norm,
                            attn_drop,
                            proj_drop,
                            dp[i],
                        )
                        for i in range(dec_depths[s])
                    )
                )
            self.out_channels = dec_channels[0]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Truncated-normal (std 0.02) init for Linear weights, zero biases.

        Convolution weights keep the He-style init from ``SubMConv3d``; norm layers keep
        their PyTorch defaults.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _prepare(self, data: dict) -> Point:
        """Validate the input dict and turn it into a ``Point`` with ``coord``, ``feat``,
        ``grid_coord`` and a contiguous ``batch`` vector."""
        if "coord" not in data or "feat" not in data:
            raise KeyError("data must contain 'coord' and 'feat'")

        coord = data["coord"]
        feat = data["feat"]
        self._validate_coord_and_feat(coord, feat)
        batch = self._prepare_batch(data, coord)
        grid_coord = self._prepare_grid_coord(data, coord, batch)
        return Point(coord=coord, feat=feat, grid_coord=grid_coord, batch=batch)

    def _validate_coord_and_feat(self, coord: torch.Tensor, feat: torch.Tensor) -> None:
        """Check shapes, channel count, emptiness and device agreement of coord / feat."""
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError("coord must have shape (N, 3)")
        if feat.ndim != 2 or feat.shape[0] != coord.shape[0]:
            raise ValueError("feat must have shape (N, C) with the same leading dimension as coord")
        if feat.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {feat.shape[1]}")
        if coord.shape[0] == 0:
            raise ValueError("Empty point clouds are not supported")
        if coord.device != feat.device:
            raise ValueError("coord and feat must be on the same device")

    @staticmethod
    def _prepare_batch(data: dict, coord: torch.Tensor) -> torch.Tensor:
        """Return a ``(N,)`` long tensor of cloud ids.

        Accepts either an explicit ``batch`` vector or cumulative ``offset`` values (e.g.
        ``[6000, 9500]`` for two clouds of 6000 and 3500 points). Empty clouds are rejected.
        Ids are re-mapped to the contiguous range ``0..B-1`` (order-preserving), so
        arbitrary non-negative ids are allowed and the packed hash keys stay small.
        """
        if "batch" in data:
            batch = data["batch"].to(device=coord.device, dtype=torch.long)
            if batch.ndim != 1 or batch.shape[0] != coord.shape[0]:
                raise ValueError("batch must have shape (N,)")
            if torch.any(batch < 0):
                raise ValueError("batch IDs must be non-negative")
        elif "offset" in data:
            off = data["offset"].to(device=coord.device, dtype=torch.long)
            if off.ndim != 1 or off.numel() == 0:
                raise ValueError("offset must be a nonempty 1D tensor")
            if off[-1] != coord.shape[0]:
                raise ValueError("offset must end at the total number of points")
            counts = torch.diff(off, prepend=off.new_zeros(1))
            if torch.any(counts <= 0):
                raise ValueError("offset must be strictly increasing; empty clouds are not supported")
            batch = torch.repeat_interleave(torch.arange(off.numel(), device=coord.device), counts)
        else:
            raise KeyError("data must contain either 'batch' or 'offset'")

        if torch.numel(batch) == 0:
            raise ValueError("batch cannot be empty")

        _, batch = torch.unique(batch, sorted=True, return_inverse=True)
        return batch

    @staticmethod
    def _prepare_grid_coord(data: dict, coord: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Return ``(N, 3)`` long voxel indices.

        Uses ``data["grid_coord"]`` when given. Otherwise voxelizes ``coord`` with
        ``data["grid_size"]`` relative to each cloud's own minimum corner, so every cloud
        starts at voxel 0. Note that several points may share a voxel in that case.
        """
        if "grid_coord" in data:
            grid_coord = data["grid_coord"].to(device=coord.device, dtype=torch.long)
            if grid_coord.shape != coord.shape:
                raise ValueError("grid_coord must have shape (N, 3)")
            return grid_coord

        if "grid_size" not in data:
            raise KeyError("data must contain either 'grid_coord' or 'grid_size'")
        grid_size = float(data["grid_size"])
        if not torch.isfinite(torch.tensor(grid_size)) or grid_size <= 0:
            raise ValueError("grid_size must be a positive finite number")
        num_batches = int(batch.max()) + 1
        mn = torch.full((num_batches, 3), float("inf"), device=coord.device, dtype=coord.dtype)
        mn = mn.scatter_reduce(0, batch[:, None].expand(-1, 3), coord, "amin")  # per-cloud minimum corner
        # small epsilon guards against float rounding for points exactly on voxel boundaries
        return torch.floor((coord - mn[batch]) / grid_size + 1e-4).long()

    def forward(self, data: dict) -> Point:
        """Run the network.

        Args:
            data: Input dict (see the module docstring): ``coord``, ``feat``, and either
                ``grid_coord`` or ``grid_size``, and either ``batch`` or ``offset``.

        Returns:
            ``Point`` whose ``feat`` is ``(N, out_channels)`` in the input point order
            (decoder) or the coarsest-stage features (``enc_mode=True``). ``encoder_stages``
            lists, for each encoder stage, a ``Point`` with that stage's ``feat``,
            ``coord``, ``grid_coord``, ``batch`` and ``pooling_inverse`` (``None`` for the
            first stage); the entries reference the tensors as they were after the
            stage's last block and are not modified later.

        Steps: (1) stem conv on the finest voxels, (2) for each stage pool (except the
        first), serialize, build the 3x3x3 neighbour table, run the blocks, (3) unless
        ``enc_mode``, walk back up with unpooling + decoder blocks.
        """
        point = self._prepare(data)
        with torch.no_grad():
            grid_bound = 1 << self.serial_depth
            nbr_stem = build_neighbors(point.grid_coord, point.batch, self.stem_kernel, grid_bound)

        point.feat = F.gelu(self.embed_norm(self.embed_conv(point.feat, nbr_stem)))
        encoder_stages = []

        for s in range(self.num_stages):
            if s > 0:
                point = self.pools[s - 1](point)
            # serialization and neighbour tables depend on the stage's resolution, so they are
            # recomputed per stage and then shared by all blocks of that stage
            serialize(point, self.order, self.serial_depth, self.shuffle_orders and self.training)
            point.nbr = build_neighbors(point.grid_coord, point.batch, 3, grid_bound)
            for blk in self.enc[s]:
                point = blk(point)
            encoder_stages.append(
                Point(
                    feat=point.feat,
                    coord=point.coord,
                    grid_coord=point.grid_coord,
                    batch=point.batch,
                    pooling_inverse=point.pooling_inverse if s > 0 else None,
                )
            )

        point.encoder_stages = encoder_stages
        if not self.enc_mode:
            for up, blocks in zip(self.dec_up, self.dec):
                point = up(point)
                for blk in blocks:
                    point = blk(point)
        return point


if __name__ == "__main__":
    # Smoke test: two random voxel clouds, encoder-only and full encoder+decoder, plus a
    # backward pass and a parameter count.
    from components.utils.device import get_device

    torch.manual_seed(0)
    gs = 0.02  # voxel size used to derive float coordinates from integer voxel indices

    device = get_device()

    def make_cloud(n):
        # unique integer voxels in a 50^3 grid (one point per voxel) and matching coordinates
        g = torch.unique(torch.randint(0, 50, (n, 3)), dim=0).to(device)
        return g.float() * gs, g

    clouds = [make_cloud(6000), make_cloud(3500)]
    coord = torch.cat([c for c, _ in clouds])
    batch = torch.cat([torch.full((len(c),), i) for i, (c, _) in enumerate(clouds)])
    grid_coord = torch.cat([g for _, g in clouds])
    data = dict(coord=coord, feat=torch.cat([coord, torch.rand_like(coord)], 1), batch=batch, grid_coord=grid_coord)

    # small configuration: 4 stages, patch size 128
    config = PointTransformerV3Config(
        in_channels=6,
        enc_depths=(2, 2, 2, 2),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(128,) * 4,
        stride=(2, 2, 2),
        dec_depths=(1, 1, 1),
        dec_channels=(32, 64, 128),
        dec_num_head=(2, 4, 8),
        dec_patch_size=(128,) * 3,
    )
    enc = PointTransformerV3Encoder(replace(config, enc_mode=True)).to(device)
    out = enc(data)
    print("encoder:", tuple(out.feat.shape), "points per cloud ->", out.batch.bincount().tolist())

    model = PointTransformerV3Encoder(config).to(device)
    out = model(data)
    print("encoder+decoder:", tuple(out.feat.cpu().shape))
    out.feat.mean().backward()
    print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
