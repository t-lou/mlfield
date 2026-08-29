from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams
from components.mmperc.backbone.tiny_bev_backbone import TinyBEVBackbone
from components.mmperc.scatter.scatter import scatter_to_bev
from components.mmperc.voxelizer.pointpillar_lite import PointpillarLite
from components.utils.calibration import load_sensor_calibration


def _morton_code(x: Tensor, y: Tensor) -> Tensor:
    """Return a 2D Morton/Z-order key for non-negative integer coordinates."""
    code = torch.zeros_like(x)
    for bit in range(16):
        code |= ((x >> bit) & 1) << (2 * bit)
        code |= ((y >> bit) & 1) << (2 * bit + 1)
    return code


class PointTransformerBlock(nn.Module):
    """Local pre-norm transformer block used on serialized point tokens."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        normalized = self.norm_attn(tokens)
        attended, _ = self.attn(normalized, normalized, normalized, key_padding_mask=key_padding_mask)
        tokens = tokens + attended
        return tokens + self.ffn(tokens)


class PointTransformerV3BEV(nn.Module):
    """PTv3-inspired point encoder with a dense BEV output contract.

    This follows the useful PTv3 design principles without pretending to be a
    drop-in reproduction of the official implementation: points are encoded in
    local pillars, serialized in Morton order, processed in fixed-size windows,
    and pooled back to their pillars before dense BEV decoding.

    Input: ``{sensor_name: (B, N, >=4)}``.
    Output: ``(B, params.bev_params.bev_channels, BEV_H/2, BEV_W/2)``.
    """

    def __init__(
        self,
        params: MmpercParams,
        sensor_names: list[str],
        num_heads: int = 8,
        depth: int = 3,
        window_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dim = params.bev_params.bev_channels
        if dim % num_heads != 0:
            raise ValueError(f"bev_channels ({dim}) must be divisible by num_heads ({num_heads})")
        if window_size < 1:
            raise ValueError("window_size must be positive")

        self.sensor_names = list(sensor_names)
        self.window_size = window_size
        self.bev_h = params.bev_params.bev_h
        self.bev_w = params.bev_params.bev_w
        self.voxelizer = PointpillarLite(params=params)
        self.calibrations = {
            name: load_sensor_calibration(Path(params.path_calibration), sensor_name=name, sensor_type="camera")
            for name in self.sensor_names
        }

        self.point_input = nn.Sequential(
            nn.Linear(9, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.point_blocks = nn.ModuleList([PointTransformerBlock(dim, num_heads, dropout) for _ in range(depth)])
        self.pillar_pool = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.pillar_position = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.pillar_blocks = nn.ModuleList([PointTransformerBlock(dim, num_heads, dropout) for _ in range(depth)])
        self.backbone = TinyBEVBackbone(params=params, in_channels=dim, out_channels=dim)

    @staticmethod
    def transform_points_to_vehicle(calibration, points: Tensor) -> Tensor:
        if points.ndim == 2:
            points = points.unsqueeze(0)
        if points.ndim != 3 or points.shape[-1] < 3:
            raise ValueError(f"Expected points of shape (B, N, >=3), got {tuple(points.shape)}")
        xyz = points[..., :3]
        ones = torch.ones(*xyz.shape[:-1], 1, device=points.device, dtype=points.dtype)
        xyz_h = torch.cat([xyz, ones], dim=-1)
        transform = torch.as_tensor(calibration.pose.vehicle_from_sensor, device=points.device, dtype=points.dtype)
        vehicle_xyz = (xyz_h @ transform.T)[..., :3]
        return torch.cat([vehicle_xyz, points[..., 3:]], dim=-1)

    def _serialize(
        self, pillar_features: Tensor, pillar_coords: Tensor, pillar_valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_pillars, dim = pillar_features.shape
        # argsort below runs per batch row (dim=1), so batches are already
        # sorted independently -- no batch offset needs to be folded into
        # the key itself.
        keys = _morton_code(pillar_coords[..., 0].long(), pillar_coords[..., 1].long())
        keys = keys.masked_fill(~pillar_valid, torch.iinfo(keys.dtype).max)
        order = torch.argsort(keys, dim=1)
        return (
            pillar_features.gather(1, order.unsqueeze(-1).expand(-1, -1, dim)),
            pillar_valid.gather(1, order),
            order,
        )

    def _encode(self, points: Tensor) -> Tensor:
        voxels = self.voxelizer(points)
        pillar_points = self.point_input(voxels["pillars"])
        point_valid = voxels["pillars"].abs().sum(dim=-1) > 0
        point_padding = ~point_valid
        # A pillar with zero real points would otherwise have an all-True
        # padding row, which makes every attention score -inf and softmax
        # NaN. Force one token unmasked -- its output is discarded later
        # since point_valid (not point_padding) drives the pooling weights.
        point_padding[..., 0] = False

        # nn.MultiheadAttention (batch_first=True) only accepts 3D (batch,
        # seq, dim) input. `pillar_points` is (B, P, M, dim), so each
        # pillar's M points need to be flattened into the batch dimension
        # before attending over them, then unflattened afterward. The mask
        # was already being reshaped this way below; the features were not,
        # which would raise a shape error the first time this ran.
        b, p, m, d = pillar_points.shape
        pillar_points = pillar_points.reshape(b * p, m, d)
        point_padding_flat = point_padding.reshape(b * p, m)
        for block in self.point_blocks:
            pillar_points = block(pillar_points, key_padding_mask=point_padding_flat)
        pillar_points = pillar_points.reshape(b, p, m, d)

        point_weights = point_valid.to(pillar_points.dtype).unsqueeze(-1)
        pooled = (pillar_points * point_weights).sum(dim=2) / point_weights.sum(dim=2).clamp_min(1.0)
        pooled = self.pillar_pool(pooled)
        coordinates = voxels["pillar_coords"].to(dtype=pooled.dtype)
        coordinates = coordinates / coordinates.new_tensor([self.bev_w, self.bev_h])
        pooled = pooled + self.pillar_position(coordinates)

        # Pillar validity must come from the raw points (point_valid, above),
        # not from `pooled`: LayerNorm's affine bias in pillar_pool and the
        # position embedding both add a nonzero value to every pillar,
        # including empty ones, so by the time a pillar reaches `pooled` an
        # all-zero (empty) pillar no longer looks like all-zeros.
        pillar_valid = point_valid.any(dim=-1)

        serialized, valid, order = self._serialize(pooled, voxels["pillar_coords"], pillar_valid)
        batch_size, num_pillars, dim = serialized.shape
        padding = ~valid

        # Windows never attend to each other, so instead of looping over
        # them in Python (issuing depth * num_windows small attention calls
        # per forward pass) we fold the window dimension into the batch
        # dimension and run each block once over every window at once.
        window = self.window_size
        num_windows = -(-num_pillars // window)  # ceil division
        pad_amount = num_windows * window - num_pillars
        if pad_amount:
            serialized = torch.cat([serialized, serialized.new_zeros(batch_size, pad_amount, dim)], dim=1)
            padding = torch.cat([padding, padding.new_ones(batch_size, pad_amount)], dim=1)

        windowed = serialized.view(batch_size * num_windows, window, dim)
        window_padding = padding.view(batch_size * num_windows, window).clone()

        # A window can end up entirely padding once real pillars run out
        # (they're sorted to the front by _serialize), or from the padding
        # added above. An all-True key_padding_mask row makes every
        # attention score -inf for that item, and softmax over an all -inf
        # row is NaN. Force one token unmasked in that case -- its output is
        # discarded anyway since the whole window is invalid, but this stops
        # the NaN from ever being written into `serialized`.
        fully_padded = window_padding.all(dim=1)
        window_padding[fully_padded, 0] = False

        for block in self.pillar_blocks:
            windowed = block(windowed, key_padding_mask=window_padding)

        serialized = windowed.view(batch_size, num_windows * window, dim)[:, :num_pillars]

        inverse = torch.argsort(order, dim=1)
        pooled = serialized.gather(1, inverse.unsqueeze(-1).expand(-1, -1, dim))
        # Zero out invalid pillars before scattering. pillar_pool /
        # pillar_position (and, transiently, any NaN-guarded window above)
        # can leave non-zero values in pillars that have no real points, and
        # every invalid pillar shares coordinate (0, 0) in PointpillarLite's
        # padded output -- left unmasked, that would land on top of whatever
        # real feature belongs at BEV cell (0, 0).
        pooled = pooled * pillar_valid.unsqueeze(-1).to(pooled.dtype)
        bev = scatter_to_bev(pooled, voxels["pillar_coords"], self.bev_h, self.bev_w)
        return self.backbone(bev)

    def forward(self, points_by_sensor: dict[str, Tensor]) -> Tensor:
        transformed = [
            self.transform_points_to_vehicle(self.calibrations[name], points)
            for name, points in points_by_sensor.items()
        ]
        if not transformed:
            raise ValueError("No point clouds were provided to PointTransformerV3BEV")
        return self._encode(torch.cat(transformed, dim=1))
