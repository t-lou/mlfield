import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mask_tokens(tokens: Tensor, mask_ratio: float) -> Tensor:
    """Randomly hide token content while retaining its positions."""
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in the range [0, 1)")
    if mask_ratio == 0.0:
        return tokens

    keep = torch.rand(tokens.shape[:2], device=tokens.device) >= mask_ratio
    keep[:, 0] = True
    return tokens * keep.unsqueeze(-1).to(dtype=tokens.dtype)


def _mask_bev(bev: Tensor, mask_ratio: float) -> Tensor:
    B, _, H, W = bev.shape
    masked = _mask_tokens(bev.flatten(2).transpose(1, 2), mask_ratio)
    return masked.transpose(1, 2).reshape(B, -1, H, W)


class LatentPredictor(nn.Module):
    """Predict target latent slots from the masked context slots."""

    def __init__(self, dim: int, hidden_mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, context_latent: Tensor) -> Tensor:
        return self.net(context_latent)


def sigreg_loss(embeddings: Tensor, num_slices: int = 64) -> Tensor:
    """Penalize deviations from unit-normal projected embeddings.

    This is a compact sliced-normality regularizer inspired by SIGReg. It uses
    random normalized projections and matches their characteristic function to
    that of a standard normal distribution.
    """
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape (B, N, D)")
    flat = F.normalize(embeddings.reshape(-1, embeddings.shape[-1]), dim=-1)
    directions = torch.randn(num_slices, flat.shape[-1], device=flat.device, dtype=flat.dtype)
    directions = F.normalize(directions, dim=-1)
    projected = flat @ directions.transpose(0, 1)
    projected = (projected - projected.mean(dim=0, keepdim=True)) / projected.std(dim=0, keepdim=True).clamp_min(1e-4)

    frequencies = torch.arange(1, 8, device=flat.device, dtype=flat.dtype).view(-1, 1)
    characteristic = torch.exp(1j * projected.to(torch.complex64).unsqueeze(0) * frequencies)
    empirical = characteristic.mean(dim=1)
    target = torch.exp(-0.5 * frequencies.square()).to(empirical.dtype)
    return (empirical.real - target.real).square().mean() + empirical.imag.square().mean()


class LeJEPA(nn.Module):
    """End-to-end LeJEPA objective around a multimodal JEPA encoder."""

    def __init__(self, encoder: nn.Module, dim: int, sigreg_weight: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = LatentPredictor(dim)
        self.sigreg_weight = sigreg_weight

    def forward(
        self,
        lidar_points: dict[str, Tensor],
        camera_images: dict[str, Tensor],
        cam_meta: dict[str, dict] | None = None,
        can_tokens: Tensor | None = None,
        mask_ratio: float = 0.5,
    ) -> dict[str, Tensor]:
        context = self.encoder.encode_modalities(lidar_points, camera_images, cam_meta, can_tokens)
        target = {name: value for name, value in context.items()}

        context_lidar = _mask_bev(context["lidar"], mask_ratio)
        context_cameras = {name: _mask_tokens(tokens, mask_ratio) for name, tokens in context["cameras"].items()}
        context_latent = self.encoder.fuse(context_lidar, context_cameras, context["can"])
        target_latent = self.encoder.fuse(target["lidar"], target["cameras"], target["can"])

        prediction = self.predictor(context_latent)
        prediction = F.normalize(prediction, dim=-1)
        target_latent = F.normalize(target_latent, dim=-1)
        prediction_loss = 2.0 - 2.0 * (prediction * target_latent).sum(dim=-1).mean()
        regularization = sigreg_loss(torch.cat([prediction, target_latent], dim=0))
        return {
            "loss": prediction_loss + self.sigreg_weight * regularization,
            "prediction_loss": prediction_loss,
            "sigreg_loss": regularization,
            "context_latent": context_latent,
            "target_latent": target_latent,
        }


class JEPAEncoder(nn.Module):
    """
    Multimodal JEPA encoder:
        - LiDAR (multi-sensor)
        - Cameras (multi-camera)
        - CAN (optional)
    Produces:
        - shared latent representation for SSL
        - optional BEV projection for fine-tuning
    """

    def __init__(
        self,
        lidar_encoder: nn.Module,
        camera_encoder: nn.Module,
        fusion_module: nn.Module,
        bev_projector: nn.Module = None,
    ):
        super().__init__()
        self.lidar_encoder = lidar_encoder
        self.camera_encoder = camera_encoder
        self.fusion_module = fusion_module
        self.bev_projector = bev_projector  # used only in fine-tuning

    def forward_ssl(self, lidar_points_list, camera_images_list, cam_meta_list, can_bus=None):
        """
        SSL forward pass (no heads, no BEV projection).
        Uses ALL sensors.

        Args:
            lidar_points_list: list of LiDAR point clouds (one per sensor)
            camera_images_list: list of images (one per camera)
            cam_meta_list: list of calibration dicts (intrinsics/extrinsics)
            can_bus: optional CAN history

        Returns:
            latent: JEPA shared latent representation
        """

        encoded = self.encode_modalities(lidar_points_list, camera_images_list, cam_meta_list, can_bus)
        return self.fuse(encoded["lidar"], encoded["cameras"], encoded["can"])

    def encode_modalities(self, lidar_points, camera_images, cam_meta=None, can_tokens=None):
        lidar_bev = self.lidar_encoder(lidar_points)
        camera_tokens, _, _ = self.camera_encoder(camera_images, cam_meta)
        return {"lidar": lidar_bev, "cameras": camera_tokens, "can": can_tokens}

    def fuse(self, lidar_bev, camera_tokens, can_tokens=None):
        return self.fusion_module(lidar_bev, camera_tokens, can_tokens)

    def forward_finetune(self, latent, mode="front"):
        """
        Fine-tuning forward pass:
            - BEV projection
            - detection/segmentation/e2e heads

        Args:
            latent: JEPA latent from SSL
            mode: "front" or "all" (for datasets with full labels)

        Returns:
            bev_features: BEV tensor for heads
        """

        if self.bev_projector is None:
            raise ValueError("BEV projector required for fine-tuning")

        bev_features = self.bev_projector(latent, mode=mode)
        return bev_features
