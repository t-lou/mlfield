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

    def _serialize(self, pillar_features: Tensor, pillar_coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_pillars, dim = pillar_features.shape
        valid = pillar_features.abs().sum(dim=-1) > 0
        batch_ids = torch.arange(batch_size, device=pillar_features.device).view(-1, 1).expand_as(valid)
        keys = _morton_code(pillar_coords[..., 0].long(), pillar_coords[..., 1].long())
        keys = keys + batch_ids * (1 << 32)
        keys = keys.masked_fill(~valid, torch.iinfo(keys.dtype).max)
        order = torch.argsort(keys, dim=1)
        return (
            pillar_features.gather(1, order.unsqueeze(-1).expand(-1, -1, dim)),
            valid.gather(1, order),
            order,
        )

    def _encode(self, points: Tensor) -> Tensor:
        voxels = self.voxelizer(points)
        pillar_points = self.point_input(voxels["pillars"])
        point_valid = voxels["pillars"].abs().sum(dim=-1) > 0
        point_padding = ~point_valid
        point_padding[..., 0] = False
        for block in self.point_blocks:
            pillar_points = block(pillar_points, key_padding_mask=point_padding.reshape(-1, point_padding.shape[-1]))
        point_weights = point_valid.to(pillar_points.dtype).unsqueeze(-1)
        pooled = (pillar_points * point_weights).sum(dim=2) / point_weights.sum(dim=2).clamp_min(1.0)
        pooled = self.pillar_pool(pooled)
        coordinates = voxels["pillar_coords"].to(dtype=pooled.dtype)
        coordinates = coordinates / coordinates.new_tensor([self.bev_w, self.bev_h])
        pooled = pooled + self.pillar_position(coordinates)

        serialized, valid, order = self._serialize(pooled, voxels["pillar_coords"])
        batch_size, num_pillars, dim = serialized.shape
        padding = ~valid
        for start in range(0, num_pillars, self.window_size):
            stop = min(start + self.window_size, num_pillars)
            serialized[:, start:stop] = self.pillar_blocks[0](
                serialized[:, start:stop], key_padding_mask=padding[:, start:stop]
            )
            for block in self.pillar_blocks[1:]:
                serialized[:, start:stop] = block(serialized[:, start:stop], key_padding_mask=padding[:, start:stop])

        inverse = torch.argsort(order, dim=1)
        pooled = serialized.gather(1, inverse.unsqueeze(-1).expand(-1, -1, dim))
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
