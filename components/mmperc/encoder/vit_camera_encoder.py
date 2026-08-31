"""ViT-based camera encoder for multimodal perception.

Designed to replace CNN-based encoders (e.g., TinyCameraEncoder) with
Vision Transformer backbone. Supports:
  - Multi-scale feature extraction for dense prediction heads
  - SSL pretraining (DINO/JEPA style)
  - Easy swapping with TinyCameraEncoder in perception pipelines
"""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams
from components.utils.calibration import load_sensor_calibration
from components.utils.logger import logger
from components.vit.vit_encoder import VitEncoder


class ViTCameraEncoder(nn.Module):
    """Vision Transformer-based camera encoder for multimodal perception.

    Converts an RGB image:
        (B, 3, H, W)

    Into a sequence of camera tokens:
        (B, N_cam, C)
    and a feature map:
        (B, C, H', W')

    where:
        - N_cam = H' * W' (spatial tokens after patch embedding)
        - C = embed_dim (transformer embedding dimension)

    The encoder extracts multi-scale features from intermediate transformer
    blocks, enabling dense prediction heads for semantic segmentation and
    bbox detection.

    Args:
        params: MmpercParams configuration
        sensor_name: Camera sensor name for calibration loading
        patch_size: Patch size for ViT (default 16)
        embed_dim: Transformer embedding dimension (default 384)
        depth: Number of transformer blocks (default 12)
        num_heads: Number of attention heads (default 6)
        mlp_ratio: MLP hidden dimension ratio (default 4.0)
        drop_path_rate: Stochastic depth rate (default 0.1)
        extract_layers: Indices of transformer blocks to extract features from
                       (default [3, 6, 9] for 3-scale hierarchy)
    """

    def __init__(
        self,
        params: MmpercParams,
        sensor_name: str = "front_center",
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        extract_layers: list[int] | None = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_scale = params.image_scale

        # Load camera calibration for geometric features
        calibration_path = Path(params.path_calibration)
        self.camera_calibration = load_sensor_calibration(
            calibration_path,
            sensor_name=sensor_name,
            sensor_type="camera",
        )

        K = self.camera_calibration.intrinsic.astype(np.float32)
        self.fx = float(K[0, 0] * self.image_scale)
        self.fy = float(K[1, 1] * self.image_scale)
        self.cx = float(K[0, 2] * self.image_scale)
        self.cy = float(K[1, 2] * self.image_scale)
        self._camera_geometry_cache: dict[tuple[int, int, int, int], Tensor] = {}

        logger.info(f"ViTCameraEncoder: camera_calibration={self.camera_calibration}")

        pose_matrix = self.camera_calibration.pose.sensor_from_vehicle.astype(np.float32)
        self.register_buffer("cam_pose_vector", torch.from_numpy(pose_matrix.reshape(-1)))

        # ViT backbone
        self.vit = VitEncoder(
            base_res=224,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            qkv_bias=False,
            add_cls_token=True,
        )

        # Configure layer extraction for multi-scale features
        if extract_layers is None:
            # Default: extract from ~25%, ~50%, ~100% of depth for 3-scale pyramid
            extract_layers = [depth // 4, depth // 2, depth]
        self.extract_layers = extract_layers

        # Lateral projections to normalize intermediate features to embed_dim
        self.lateral_projs = nn.ModuleDict()
        for i, layer_idx in enumerate(extract_layers):
            self.lateral_projs[f"layer_{i}"] = nn.Linear(embed_dim, embed_dim, bias=False)

        # Geometric feature projection
        self.camera_pos_project = nn.Conv2d(2, embed_dim, kernel_size=1, bias=False)
        self.camera_pos_scale = nn.Parameter(torch.tensor(1.0))

        # LayerNorm for token normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Encode RGB image into tokens and multi-scale feature maps.

        Args:
            x: RGB image tensor of shape (B, 3, H, W)

        Returns:
            tokens: (B, N_cam, embed_dim)
                    Flattened spatial tokens (without CLS token)
            feat:   (B, embed_dim, H', W')
                    Feature map at finest extracted scale
            skip_feats: dict[str, Tensor]
                        Multi-scale features:
                        - "s0": Finest scale (1/patch_size resolution)
                        - "s1": Mid scale
                        - "s2": Coarsest scale
        """
        B = x.shape[0]
        H_img, W_img = x.shape[2:]
        H_patch = H_img // (self.patch_size * self.image_scale)
        W_patch = W_img // (self.patch_size * self.image_scale)

        # Tokenize: patch embedding + positional encoding
        tokens = self.vit._tokenize(x)  # (B, 1 + H'*W', C) with CLS token

        # Extract intermediate features from specified layers
        intermediate_features = {}
        x_feat = tokens

        for i, blk in enumerate(self.vit.blocks):
            x_feat = blk(x_feat)
            if (i + 1) in self.extract_layers:
                layer_idx = self.extract_layers.index(i + 1)
                intermediate_features[f"layer_{layer_idx}"] = x_feat.clone()

        # Apply final normalization
        x_feat = self.vit.norm(x_feat)

        # Remove CLS token for spatial operations
        tokens_spatial = x_feat[:, 1:, :]  # (B, H'*W', C)

        # Add geometric features (normalized image coordinates)
        geom = self._camera_geometry((H_patch, W_patch), (H_img, W_img), x.device, x.dtype)
        geom_feat = self.camera_pos_project(geom)
        tokens_spatial = tokens_spatial + self.camera_pos_scale * geom_feat.flatten(2).transpose(1, 2)

        # Normalize tokens
        tokens = self.norm(tokens_spatial)

        # Reconstruct spatial feature maps at each scale
        skip_feats = {}
        for i, layer_idx in enumerate(self.extract_layers):
            feat_tokens = intermediate_features[f"layer_{i}"][:, 1:, :]  # Remove CLS
            proj_feat = self.lateral_projs[f"layer_{i}"](feat_tokens)

            # Reshape to spatial dimensions
            # Feature resolution: H_patch / (2^layer_scale), W_patch / (2^layer_scale)
            scale_factor = 2 ** (len(self.extract_layers) - 1 - i)
            H_feat = H_patch // scale_factor
            W_feat = W_patch // scale_factor

            # Reshape to (B, H_feat, W_feat, C) then to (B, C, H_feat, W_feat)
            feat_spatial = proj_feat.reshape(B, H_feat, W_feat, self.embed_dim)
            feat_spatial = feat_spatial.permute(0, 3, 1, 2)

            skip_feats[f"s{i}"] = feat_spatial

        # Use finest scale as main feature map
        feat = tokens_spatial.reshape(B, H_patch, W_patch, self.embed_dim)
        feat = feat.permute(0, 3, 1, 2)

        return tokens, feat, skip_feats

    def _camera_geometry(
        self,
        feat_hw: tuple[int, int],
        img_hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Generate normalized image coordinates as geometric features.

        Caches computed geometries to avoid redundant computation.

        Args:
            feat_hw: (H_feat, W_feat) - feature map spatial dimensions
            img_hw: (H_img, W_img) - input image spatial dimensions
            device: Torch device
            dtype: Torch dtype

        Returns:
            Tensor of shape (1, 2, H_feat, W_feat) with normalized (u, v) coordinates
        """
        H_feat, W_feat = feat_hw
        H_img, W_img = img_hw
        cache_key = (H_feat, W_feat, H_img, W_img)

        if cache_key in self._camera_geometry_cache:
            cached = self._camera_geometry_cache[cache_key]
            if cached.device == device and cached.dtype == dtype:
                return cached
            return cached.to(device=device, dtype=dtype)

        # Sample centers of feature cells in image space
        y = torch.linspace(0.5 * H_img / H_feat, H_img - 0.5 * H_img / H_feat, H_feat, device=device, dtype=dtype)
        x = torch.linspace(0.5 * W_img / W_feat, W_img - 0.5 * W_img / W_feat, W_feat, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        # Normalize to camera frame (undistorted)
        u = (grid_x - self.cx) / self.fx
        v = (grid_y - self.cy) / self.fy

        geom = torch.stack([u, v], dim=0).unsqueeze(0)
        self._camera_geometry_cache[cache_key] = geom.detach().to(device=device, dtype=dtype)

        return geom

    def load_pretrained(self, pretrained_path: str, ssl_type: str = "dino") -> None:
        """Load pretrained weights from SSL models (DINO, iJEPA, MAE).

        Args:
            pretrained_path: Path to pretrained checkpoint
            ssl_type: Type of SSL pretraining ("dino", "jepa", "mae")
        """
        logger.info(f"Loading pretrained weights from {pretrained_path} (ssl_type={ssl_type})")
        checkpoint = torch.load(pretrained_path, map_location="cpu")

        if ssl_type == "dino":
            # DINO checkpoints store backbone in 'student' or 'model'
            if "student" in checkpoint:
                state_dict = checkpoint["student"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        elif ssl_type == "jepa":
            # iJEPA checkpoints typically have 'target_encoder' or 'encoder'
            if "target_encoder" in checkpoint:
                state_dict = checkpoint["target_encoder"]
            else:
                state_dict = checkpoint
        elif ssl_type == "mae":
            # MAE checkpoints usually have 'model' or direct state
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError(f"Unknown ssl_type: {ssl_type}")

        # Load only VitEncoder weights
        vit_state = {}
        for key, val in state_dict.items():
            if key.startswith("vit.") or key.startswith("backbone.vit."):
                # Strip prefix
                clean_key = key.replace("backbone.", "")
                vit_state[clean_key] = val

        self.vit.load_state_dict(vit_state, strict=False)
        logger.info("Pretrained weights loaded successfully")


def _smoke_test():
    """Smoke test for ViTCameraEncoder."""
    B, C_in, H, W = 2, 3, 256, 256
    x = torch.rand(B, C_in, H, W)

    params = MmpercParams()
    encoder = ViTCameraEncoder(params=params)

    tokens, feat, skip_feats = encoder(x)

    # Check output shapes
    assert tokens.shape[0] == B and tokens.shape[2] == encoder.embed_dim, f"Got {tokens.shape}"
    assert feat.shape[0] == B and feat.shape[1] == encoder.embed_dim, f"Got {feat.shape}"
    assert "s0" in skip_feats, "Missing s0 in skip_feats"

    print("✓ ViTCameraEncoder smoke test passed")
    print(f"  Input: {x.shape}")
    print(f"  Tokens: {tokens.shape}")
    print(f"  Feat: {feat.shape}")
    print(f"  Skip feats: {[(k, v.shape) for k, v in skip_feats.items()]}")


if __name__ == "__main__":
    _smoke_test()
