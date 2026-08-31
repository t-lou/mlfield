"""
Abstract teacher models for knowledge distillation.

This module provides a generic interface for teacher models used in knowledge distillation.
Supports multiple pretrained models: MAE, DINO, I-JEPA, and extensible for others.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from components.utils.logger import logger


class TeacherModel(ABC):
    """
    Abstract base class for teacher models used in knowledge distillation.

    A teacher model provides high-quality features learned through self-supervised
    pretraining. The YOLO backbone uses these features as additional supervision
    signals to improve convergence and generalization.
    """

    @abstractmethod
    def to(self, device: str) -> "TeacherModel":
        """Move model to device."""
        pass

    @abstractmethod
    def eval(self) -> None:
        """Set model to evaluation mode."""
        pass

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Extract features from input images.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            features: (B, H', W', feature_dim) - spatial features
            feature_dim: Dimension of features
            scale: Downsampling factor (e.g., 16 for P4 features)
        """
        pass

    @abstractmethod
    def parameters(self):
        """Return model parameters."""
        pass

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Expected input size for the teacher model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the teacher model."""
        pass


class MAETeacher(TeacherModel):
    """
    Masked Autoencoder (MAE) as a teacher model.

    MAE learns to reconstruct masked patches, developing a strong understanding
    of spatial structure and texture. These features are particularly useful
    for object detection tasks.
    """

    def __init__(self, checkpoint_path: str | None = None, variant: str = "imagenet"):
        """
        Initialize MAE teacher model.

        Args:
            checkpoint_path: Path to MAE checkpoint
            variant: MAE variant (e.g., 'imagenet', 'imagenet_mini')
        """
        from components.vit.mae import MAE
        from components.vit.mae_defs import MAEConfig

        try:
            # Create MAE with default config
            self.mae = MAE(MAEConfig())
            self.variant = variant

            # Try to load checkpoint
            ckpt_path = None
            if checkpoint_path:
                provided_path = Path(checkpoint_path)
                if provided_path.is_absolute():
                    if provided_path.exists():
                        ckpt_path = provided_path
                else:
                    if provided_path.exists():
                        ckpt_path = provided_path
                    else:
                        # Try relative to module
                        local_root = Path(__file__).parent.parent.parent
                        candidate = local_root / provided_path
                        if candidate.exists():
                            ckpt_path = candidate

                if ckpt_path is None:
                    logger.warning(f"MAE checkpoint not found: {checkpoint_path}")
            else:
                # Try default location
                local_root = Path(__file__).parent.parent.parent
                default_path = local_root / "mae_checkpoints" / "final.pth"
                if default_path.exists():
                    ckpt_path = default_path

            if ckpt_path:
                try:
                    self.mae.load_checkpoint(path=ckpt_path, device="cpu")
                    logger.info(f"Loaded MAE teacher from {ckpt_path}")
                except Exception as e:
                    logger.warning(f"Failed to load MAE checkpoint: {e}")

            # Freeze MAE
            for param in self.mae.parameters():
                param.requires_grad = False
            self.mae.eval()

        except Exception as e:
            logger.error(f"Failed to initialize MAE teacher: {e}")
            raise

    def to(self, device: str) -> "MAETeacher":
        self.mae = self.mae.to(device)
        return self

    def eval(self) -> None:
        self.mae.eval()

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Extract MAE encoder features at intermediate resolution.

        Returns features at 1/16 scale (P4 equivalent) with 768 dimensions.
        """
        with torch.no_grad():
            # MAE encoder processes full image
            mae_input = F.interpolate(
                x, size=(self.mae.cfg.image_size, self.mae.cfg.image_size), mode="bilinear", align_corners=False
            )

            # Get encoder output (includes CLS token)
            mae_latent = self.mae.forward_encoder_full(mae_input, mask_ratio=0.0)  # (B, num_patches+1, 768)

            # Remove CLS token and reshape to spatial format
            mae_features = mae_latent[:, 1:, :]  # (B, 196, 768)

            B, N, C = mae_features.shape
            h = int(np.sqrt(N))  # 14 for 224x224 input

            # Reshape to spatial: (B, H, W, C) -> (B, C, H, W)
            features_spatial = mae_features.reshape(B, h, h, C).permute(0, 3, 1, 2)

            return features_spatial, C, 16  # 16x downsampling

    def parameters(self):
        return self.mae.parameters()

    @property
    def input_size(self) -> int:
        return self.mae.cfg.image_size

    @property
    def model_name(self) -> str:
        return f"MAE-{self.variant}"


class DINOTeacher(TeacherModel):
    """
    DINO (Vision Transformer with knowledge distillation) as a teacher model.

    DINO learns feature representations through self-supervised learning with
    knowledge distillation. It produces features at multiple scales.
    """

    def __init__(self, checkpoint_path: str | None = None, variant: str = "base"):
        """
        Initialize DINO teacher model.

        Args:
            checkpoint_path: Path to DINO checkpoint
            variant: DINO variant (e.g., 'small', 'base')
        """
        from components.vit.dino_defs import DINOConfig
        from components.vit.dino_model import DINOModel

        try:
            config = DINOConfig()
            self.dino = DINOModel(config)
            self.variant = variant

            # Try to load checkpoint
            ckpt_path = None
            if checkpoint_path:
                provided_path = Path(checkpoint_path)
                if provided_path.is_absolute():
                    if provided_path.exists():
                        ckpt_path = provided_path
                else:
                    if provided_path.exists():
                        ckpt_path = provided_path
                    else:
                        local_root = Path(__file__).parent.parent.parent
                        candidate = local_root / provided_path
                        if candidate.exists():
                            ckpt_path = candidate

                if ckpt_path is None:
                    logger.warning(f"DINO checkpoint not found: {checkpoint_path}")
            else:
                # Try default location
                local_root = Path(__file__).parent.parent.parent
                default_path = local_root / "dino_checkpoints" / "final.pth"
                if default_path.exists():
                    ckpt_path = default_path

            if ckpt_path:
                try:
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    self.dino.load_state_dict(state_dict)
                    logger.info(f"Loaded DINO teacher from {ckpt_path}")
                except Exception as e:
                    logger.warning(f"Failed to load DINO checkpoint: {e}")

            # Freeze DINO
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()

        except Exception as e:
            logger.error(f"Failed to initialize DINO teacher: {e}")
            raise

    def to(self, device: str) -> "DINOTeacher":
        self.dino = self.dino.to(device)
        return self

    def eval(self) -> None:
        self.dino.eval()

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Extract DINO features at intermediate resolution.

        DINO returns all token embeddings including CLS. We extract spatial features
        at 1/16 scale (P4 equivalent) with 384 dimensions (from ViT encoder).
        """
        with torch.no_grad():
            # Resize input to DINO's expected size
            input_size = self.dino.backbone.vit._patch_size * (self.dino.backbone.vit.pos_embed.shape[1] - 1) ** 0.5
            input_size = int(input_size)

            dino_input = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)

            # Get all token embeddings (including CLS)
            tokens = self.dino.backbone.vit.forward_full(dino_input)  # (B, num_tokens, embed_dim)

            # Remove CLS token
            tokens_spatial = tokens[:, 1:, :]  # (B, num_patches, embed_dim)

            B, N, C = tokens_spatial.shape
            h = int(np.sqrt(N))

            # Reshape to spatial: (B, H, W, C) -> (B, C, H, W)
            features_spatial = tokens_spatial.reshape(B, h, h, C).permute(0, 3, 1, 2)

            return features_spatial, C, 16  # 16x downsampling

    def parameters(self):
        return self.dino.parameters()

    @property
    def input_size(self) -> int:
        # Default ViT input size
        return 224

    @property
    def model_name(self) -> str:
        return f"DINO-{self.variant}"


class IJEPATeacher(TeacherModel):
    """
    I-JEPA (Image Joint-Embedding Predictive Architecture) as a teacher model.

    I-JEPA learns representations by predicting features of masked image regions
    from visible regions. It produces contextual features at multiple scales.
    """

    def __init__(self, checkpoint_path: str | None = None, variant: str = "base"):
        """
        Initialize I-JEPA teacher model.

        Args:
            checkpoint_path: Path to I-JEPA checkpoint
            variant: I-JEPA variant (e.g., 'base', 'large')
        """
        from components.vit.i_jepa import I_JEPA
        from components.vit.i_jepa_defs import IJEPAConfig

        try:
            config = IJEPAConfig()
            self.ijepa = I_JEPA(config)
            self.variant = variant

            # Try to load checkpoint
            ckpt_path = None
            if checkpoint_path:
                provided_path = Path(checkpoint_path)
                if provided_path.is_absolute():
                    if provided_path.exists():
                        ckpt_path = provided_path
                else:
                    if provided_path.exists():
                        ckpt_path = provided_path
                    else:
                        local_root = Path(__file__).parent.parent.parent
                        candidate = local_root / provided_path
                        if candidate.exists():
                            ckpt_path = candidate

                if ckpt_path is None:
                    logger.warning(f"I-JEPA checkpoint not found: {checkpoint_path}")
            else:
                # Try default location
                local_root = Path(__file__).parent.parent.parent
                default_path = local_root / "i_jepa_checkpoints" / "final.pth"
                if default_path.exists():
                    ckpt_path = default_path

            if ckpt_path:
                try:
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    if isinstance(state_dict, dict) and "model" in state_dict:
                        state_dict = state_dict["model"]
                    self.ijepa.load_state_dict(state_dict)
                    logger.info(f"Loaded I-JEPA teacher from {ckpt_path}")
                except Exception as e:
                    logger.warning(f"Failed to load I-JEPA checkpoint: {e}")

            # Freeze I-JEPA
            for param in self.ijepa.parameters():
                param.requires_grad = False
            self.ijepa.eval()

        except Exception as e:
            logger.error(f"Failed to initialize I-JEPA teacher: {e}")
            raise

    def to(self, device: str) -> "IJEPATeacher":
        self.ijepa = self.ijepa.to(device)
        return self

    def eval(self) -> None:
        self.ijepa.eval()

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Extract I-JEPA context encoder features.

        Returns features at 1/16 scale (P4 equivalent) with embed_dim dimensions.
        """
        with torch.no_grad():
            # Resize input to I-JEPA's expected size
            input_size = self.ijepa.cfg.image_size
            ijepa_input = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)

            # Get context encoder output (all tokens including CLS)
            tokens = self.ijepa.context_encoder.forward_full(ijepa_input)  # (B, num_tokens, embed_dim)

            # Remove CLS token
            tokens_spatial = tokens[:, 1:, :]  # (B, num_patches, embed_dim)

            B, N, C = tokens_spatial.shape
            h = int(np.sqrt(N))

            # Reshape to spatial: (B, H, W, C) -> (B, C, H, W)
            features_spatial = tokens_spatial.reshape(B, h, h, C).permute(0, 3, 1, 2)

            return features_spatial, C, 16  # 16x downsampling

    def parameters(self):
        return self.ijepa.parameters()

    @property
    def input_size(self) -> int:
        return self.ijepa.cfg.image_size

    @property
    def model_name(self) -> str:
        return f"I-JEPA-{self.variant}"


def create_teacher_model(
    teacher_name: str,
    checkpoint_path: str | None = None,
    variant: str = "base",
) -> TeacherModel | None:
    """
    Factory function to create a teacher model.

    Args:
        teacher_name: Name of the teacher model ('mae', 'dino', 'ijepa')
        checkpoint_path: Optional path to checkpoint
        variant: Model variant (e.g., 'imagenet', 'small', 'base')

    Returns:
        TeacherModel instance or None if creation fails
    """
    teacher_name_lower = teacher_name.lower()

    try:
        if teacher_name_lower == "mae":
            return MAETeacher(checkpoint_path=checkpoint_path, variant=variant)
        elif teacher_name_lower == "dino":
            return DINOTeacher(checkpoint_path=checkpoint_path, variant=variant)
        elif teacher_name_lower == "ijepa":
            return IJEPATeacher(checkpoint_path=checkpoint_path, variant=variant)
        elif teacher_name_lower == "none":
            return None
        else:
            logger.error(f"Unknown teacher model: {teacher_name}")
            logger.error("Supported: 'mae', 'dino', 'ijepa', 'none'")
            return None
    except Exception as e:
        logger.error(f"Failed to create {teacher_name} teacher: {e}")
        return None
