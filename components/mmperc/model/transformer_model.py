from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams
from components.mmperc.encoder.multi_camera_encoder import MultiCameraEncoder
from components.mmperc.encoder.point_transformer_v3 import (
    BEVGridSpec,
    PointGridSpec,
    PointTransformerV3Config,
    PointTransformerV3Encoder,
    adapt_ptv3_config_for_input,
    points_to_voxels,
    scatter_ptv3_features_to_bev,
)
from components.mmperc.encoder.vit_camera_encoder import ViTCameraEncoder
from components.mmperc.fusion.futr_fusion import FuTrFusionBlock
from components.mmperc.head.bbox2d_head import BBox3dHead
from components.mmperc.head.semantics_head import FullResSemHead
from components.utils.calibration import load_sensor_calibration
from components.vit.vit_config import ViTConfig, ViTVariant

SMALL_GPU_VIT_CONFIG = ViTConfig(variant=ViTVariant.T)
SMALL_GPU_PTV3_CONFIG = PointTransformerV3Config(
    in_channels=4,
    enc_depths=(1, 1, 1),
    enc_channels=(16, 32, 64),
    enc_num_head=(2, 4, 4),
    enc_patch_size=(128, 128, 128),
    stride=(2, 2),
    dec_depths=(1, 1),
    dec_channels=(16, 32),
    dec_num_head=(2, 4),
    dec_patch_size=(128, 128),
    mlp_ratio=2.0,
    drop_path=0.1,
    stem_kernel=3,
)


class _PointTransformerBEVEncoder(nn.Module):
    """Adapt PTv3 features to dense BEV.

    ``lidar_feature_resolutions`` can be ``None`` (the default full-resolution
    decoder output) or a sequence of encoder stage indices, for example ``(0, 2)``.
    The latter builds one BEV map per selected stage, concatenates the maps, and
    projects them back to ``bev.bev_channels``.
    """

    def __init__(
        self,
        params: MmpercParams,
        sensor_names: list[str],
        config: PointTransformerV3Config,
        lidar_feature_resolutions: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if config.in_channels != 4:
            raise ValueError("TransformerModel's PTv3 config must use in_channels=4.")
        self.sensor_names = list(sensor_names)
        self.calibrations = {
            name: load_sensor_calibration(
                Path(params.path_calibration),
                sensor_name=name,
                sensor_type="camera",
            )
            for name in self.sensor_names
        }

        bev = params.bev_params
        self.x_min, self.x_max = bev.x_range
        self.y_min, self.y_max = bev.y_range
        self.z_min, self.z_max = bev.z_range
        self.vx, self.vy, self.vz = bev.voxel_size
        self.bev_h = bev.bev_h
        self.bev_w = bev.bev_w
        self.out_h = self.bev_h // 2
        self.out_w = self.bev_w // 2
        self.bev_grid_spec = BEVGridSpec(
            x_range=(self.x_min, self.x_max),
            y_range=(self.y_min, self.y_max),
            height=self.out_h,
            width=self.out_w,
        )

        lidar_grid = PointGridSpec(
            min_xyz=(self.x_min, self.y_min, self.z_min),
            max_xyz=(self.x_max, self.y_max, self.z_max),
            voxel_size_xyz=(self.vx, self.vy, self.vz),
        )
        config = adapt_ptv3_config_for_input(config, lidar_grid)

        if lidar_feature_resolutions is not None:
            resolutions = (
                (lidar_feature_resolutions,)
                if isinstance(lidar_feature_resolutions, int)
                else tuple(lidar_feature_resolutions)
            )
            if not resolutions:
                raise ValueError("lidar_feature_resolutions cannot be empty")
            if any(stage < 0 for stage in resolutions):
                raise ValueError("lidar_feature_resolutions stage indices must be non-negative")
            if len(set(resolutions)) != len(resolutions):
                raise ValueError("lidar_feature_resolutions cannot contain duplicate stages")
            # In stage mode the encoder stages are consumed directly, so decoder
            # stages are unnecessary.
            config = replace(config, enc_mode=True)
        else:
            resolutions = None

        self.resolutions = resolutions
        self.point_transformer = PointTransformerV3Encoder(config)
        if resolutions is None:
            self.feature_proj = nn.Linear(self.point_transformer.out_channels, bev.bev_channels)
            self.resolution_proj = None
        else:
            if any(stage >= self.point_transformer.num_stages for stage in resolutions):
                raise ValueError(
                    f"Requested PTv3 stage exceeds available stages (0-{self.point_transformer.num_stages - 1})"
                )
            stage_channels = [config.enc_channels[stage] for stage in resolutions]
            self.resolution_stage_proj = nn.ModuleList(
                nn.Linear(stage_channels[i], bev.bev_channels) for i in range(len(resolutions))
            )
            self.resolution_proj = nn.Linear(
                len(resolutions) * bev.bev_channels,
                bev.bev_channels,
            )

    @staticmethod
    def _to_vehicle(calibration, points: Tensor) -> Tensor:
        xyz = points[..., :3]
        transform = torch.as_tensor(
            calibration.pose.vehicle_from_sensor,
            device=points.device,
            dtype=points.dtype,
        )
        transformed_xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        return torch.cat([transformed_xyz, points[..., 3:4]], dim=-1)

    def _voxelize(self, points: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        batch_size, num_points, _ = points.shape
        xyz = points[..., :3].reshape(-1, 3)
        raw_features = points[..., :4].reshape(-1, 4)
        batch = torch.arange(batch_size, device=points.device).repeat_interleave(num_points)
        lidar_grid = PointGridSpec(
            min_xyz=(self.x_min, self.y_min, self.z_min),
            max_xyz=(self.x_max, self.y_max, self.z_max),
            voxel_size_xyz=(self.vx, self.vy, self.vz),
        )
        nx, ny, nz = lidar_grid.dimensions
        if max(nx, ny, nz) > 1 << self.point_transformer.serial_depth:
            raise ValueError("LiDAR voxel grid exceeds the PTv3 serialization depth.")
        return points_to_voxels(
            torch.cat([xyz, raw_features[:, 3:]], dim=-1),
            batch,
            lidar_grid,
            batch_size,
            feature_dim=4,
        )

    def forward(self, points_by_sensor: dict[str, Tensor]) -> Tensor:
        if not points_by_sensor:
            raise ValueError("No point clouds were provided to TransformerModel.")

        transformed = []
        for sensor_name, points in points_by_sensor.items():
            if points.ndim != 3 or points.shape[-1] < 4:
                raise ValueError(f"Expected {sensor_name} points with shape (B, N, >=4).")
            transformed.append(self._to_vehicle(self.calibrations[sensor_name], points))
        points = transformed[0] if len(transformed) == 1 else torch.cat(transformed, dim=1)

        data, is_real = self._voxelize(points)
        encoded = self.point_transformer(data)
        num_cells = points.shape[0] * self.out_h * self.out_w

        if self.resolutions is None:
            features = self.feature_proj(encoded.feat[is_real])
            batch = encoded.batch[is_real]
            xyz = encoded.coord[is_real]
            maps = [self._scatter_bev(features, batch, xyz, num_cells)]
        else:
            stage_real = is_real
            stage_masks = [stage_real]
            for stage in encoded.encoder_stages[1:]:
                inverse = stage.pooling_inverse
                previous_real = stage_masks[-1].to(torch.float32)
                num_coarse = stage.feat.shape[0]
                coarse_real = torch.zeros(num_coarse, dtype=torch.float32, device=is_real.device)
                coarse_real.index_add_(0, inverse, previous_real)
                stage_real = coarse_real > 0
                stage_masks.append(stage_real)

            maps = []
            for i, stage_index in enumerate(self.resolutions):
                stage = encoded.encoder_stages[stage_index]
                real = stage_masks[stage_index]
                features = self.resolution_stage_proj[i](stage.feat[real])
                maps.append(self._scatter_bev(features, stage.batch[real], stage.coord[real], num_cells))

            maps = torch.cat(maps, dim=-1)
            maps = self.resolution_proj(maps)

        if self.resolutions is None:
            maps = maps[0]
        return maps.reshape(points.shape[0], self.out_h, self.out_w, -1).permute(0, 3, 1, 2).contiguous()

    def _scatter_bev(self, features: Tensor, batch: Tensor, xyz: Tensor, num_cells: int) -> Tensor:
        batch_size = num_cells // (self.out_h * self.out_w)
        result = scatter_ptv3_features_to_bev(
            features,
            batch,
            xyz,
            self.bev_grid_spec,
            batch_size,
        )
        if result.shape[0] != num_cells:
            raise ValueError("BEV scatter produced an unexpected number of batch cells")
        return result


class _ViTCameraAdapter(nn.Module):
    """Project ViT outputs to the channel and skip-feature API expected downstream."""

    def __init__(self, params: MmpercParams, sensor_name: str, out_channels: int, config: ViTConfig) -> None:
        super().__init__()
        self.encoder = ViTCameraEncoder(
            params=params,
            sensor_name=sensor_name,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            drop_path_rate=config.drop_path_rate,
            extract_layers=[config.depth],
            base_res=config.base_res,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            qkv_bias=config.qkv_bias,
            add_cls_token=config.add_cls_token,
        )
        self.out_channels = out_channels
        self.token_proj = nn.Linear(self.encoder.embed_dim, out_channels)
        self.feature_proj = nn.Conv2d(self.encoder.embed_dim, out_channels, kernel_size=1)
        self.s4_refine = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU())
        self.s2_refine = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU())
        self.register_buffer("cam_pose_vector", self.encoder.cam_pose_vector.detach().clone())

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        tokens, feat, _ = self.encoder(image)
        feat = self.feature_proj(feat)
        feat_h, feat_w = feat.shape[-2:]
        s4_size = (feat_h * 4, feat_w * 4)
        s2_size = (feat_h * 8, feat_w * 8)
        s4 = self.s4_refine(F.interpolate(feat, size=s4_size, mode="bilinear", align_corners=False))
        s2 = self.s2_refine(F.interpolate(feat, size=s2_size, mode="bilinear", align_corners=False))
        return self.token_proj(tokens), feat, {"s4": s4, "s2": s2}


class TransformerModel(nn.Module):
    """Multimodal model using PTv3 for LiDAR and ViT for camera features.

    For a reduced-capacity model, pass ``SMALL_GPU_VIT_CONFIG`` and
    ``SMALL_GPU_PTV3_CONFIG``. A 4GB GPU may also require training with batch size 1.

    ``lidar_feature_resolutions`` selects LiDAR features at PTv3 encoder stages:
    ``None`` uses the default decoded full-resolution output, an integer such as
    ``0`` uses one stage, and a sequence such as ``(0, 2)`` concatenates selected
    stage maps into the BEV channel contract.
    """

    def __init__(
        self,
        params: MmpercParams,
        vit_config: ViTConfig | None = None,
        ptv3_config: PointTransformerV3Config | None = None,
        lidar_feature_resolutions: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not params.use_lidar and not params.use_camera:
            raise ValueError("At least one of LiDAR or camera must be enabled.")

        self.vit_config = vit_config if vit_config is not None else ViTConfig()
        self.ptv3_config = ptv3_config if ptv3_config is not None else PointTransformerV3Config(in_channels=4)
        self.lidar_feature_resolutions = lidar_feature_resolutions
        self._sensor_name = "front_center"
        channels = params.bev_params.bev_channels

        assert params.image_scale == 1.0, "Image scaling not support in ViT"

        if params.use_lidar:
            self.lidar_encoder = _PointTransformerBEVEncoder(
                params,
                [self._sensor_name],
                self.ptv3_config,
                lidar_feature_resolutions=lidar_feature_resolutions,
            )
        if params.use_camera:
            camera_encoder = _ViTCameraAdapter(params, self._sensor_name, channels, self.vit_config)
            self.cam_encoder = MultiCameraEncoder({self._sensor_name: camera_encoder})

        self._use_fusion = params.use_lidar and params.use_camera
        if self._use_fusion:
            self.fusion = FuTrFusionBlock(params=params)

        if params.pred_bbox:
            self.bbox_head = BBox3dHead(channels)
        if params.pred_semantics:
            if not params.use_camera:
                raise ValueError("Semantic segmentation requires camera input.")
            self.sem_head = FullResSemHead(
                in_channels=self.cam_encoder.out_channels,
                num_classes=params.num_sem_classes,
                skip_channels=self.cam_encoder.out_channels,
            )
        self._params = params

    def forward(
        self,
        points: Tensor,
        image: Tensor,
        cam_meta: dict[str, dict] | None = None,
    ) -> dict[str, Tensor]:
        lidar_bev = None
        camera_tokens = None
        camera_feat = None
        camera_skips = None

        if self._params.use_lidar:
            lidar_bev = self.lidar_encoder({self._sensor_name: points})

        if self._params.use_camera:
            tokens, feats, skips = self.cam_encoder({self._sensor_name: image}, cam_meta)
            camera_tokens = tokens[self._sensor_name]
            camera_feat = feats[self._sensor_name]
            camera_skips = skips[self._sensor_name]

        if self._use_fusion:
            cam_hw = camera_feat.shape[-2:]
            bev_features = self.fusion(lidar_bev, camera_tokens, cam_hw=cam_hw)
        elif lidar_bev is not None:
            bev_features = lidar_bev
        else:
            bev_features = camera_feat

        outputs: dict[str, Tensor] = {}
        if hasattr(self, "bbox_head"):
            bbox = self.bbox_head(bev_features)
            outputs["bbox_heatmap"] = bbox["heatmap"]
            outputs["bbox_reg"] = bbox["reg"]
        if hasattr(self, "sem_head"):
            outputs["sem_logits"] = self.sem_head(camera_feat, camera_skips, out_size=image.shape[-2:])
        return outputs


def _smoke_test():
    """
    Smoke test for the model.
    """
    from components.definitions.mmperc_params import MmpercParams
    from components.utils.device import get_device

    device = get_device()

    params = MmpercParams()
    params.path_data = "./data/camera_lidar_semantic_bboxes.tar"
    params.image_scale = 1.0
    params.path_calibration = "./data/cams_lidars.json"

    # Create dummy inputs
    B, N, H_img, W_img = 2, 1000, 256, 256
    points = torch.rand(B, N, 4).to(device)  # (B, N, 4)
    images = torch.rand(B, 3, H_img, W_img).to(device)  # (B, 3, H_img, W_img)

    output_shapes = []
    configs = (
        ("implicit strides", replace(SMALL_GPU_PTV3_CONFIG, stride=None)),
        ("explicit strides", SMALL_GPU_PTV3_CONFIG),
    )
    for label, ptv3_config in configs:
        model = (
            TransformerModel(
                params=params,
                vit_config=SMALL_GPU_VIT_CONFIG,
                ptv3_config=ptv3_config,
            )
            .to(device)
            .eval()
        )
        with torch.no_grad():
            outputs = model(points=points, image=images)
        shapes = {key: tuple(value.shape) for key, value in outputs.items()}
        print(f"{label}: {shapes}")
        output_shapes.append(shapes)

    if output_shapes[0] != output_shapes[1]:
        raise RuntimeError(f"PTv3 implicit and explicit stride outputs differ: {output_shapes}")
    print("TransformerModel stride smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
