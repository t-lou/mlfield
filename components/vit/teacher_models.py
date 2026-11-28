"""
Abstract teacher models for knowledge distillation.

This module provides a generic interface for teacher models used in knowledge distillation.
Supports multiple pretrained models: MAE, DINO, I-JEPA, and extensible for others.

A teacher is only the (frozen) ViT encoder of an SSL model, built as a ``VitEncoder`` directly from
the checkpoint: embedding dimension, depth, patch size, resolution, CLS token and so on are read
from the weights, so any variant (T/S/M/B/L/XL, any patch size) can be used without keeping a
config in sync. The decoder / predictor / projection head of the SSL model is not needed.

Which weights are used:
- MAE:    the encoder (``encoder.*`` of the model state dict, as written by ``MAE.save_checkpoint``)
- DINO:   the EMA teacher backbone (``teacher`` entry of the session checkpoint, ``backbone.vit.*``)
- I-JEPA: the EMA target encoder (``target_encoder.*`` of the ``model`` entry, as in the I-JEPA paper)
"""

import math
import re
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F

from components.utils.logger import logger
from components.vit.vit_config import ViTConfig, ViTVariant
from components.vit.vit_encoder import VitEncoder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Accepted spellings of the ``variant`` argument. "auto" means: take it from the checkpoint.
_VARIANT_ALIASES = {
    "t": "T",
    "tiny": "T",
    "s": "S",
    "small": "S",
    "m": "M",
    "medium": "M",
    "b": "B",
    "base": "B",
    "l": "L",
    "large": "L",
    "xl": "XL",
    "h": "XL",
    "huge": "XL",
}


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
            features: (B, feature_dim, H', W') - spatial feature map
            feature_dim: Dimension of features
            scale: Downsampling factor (the patch size, e.g. 16 for P4 features)
        """
        pass

    @abstractmethod
    def parameters(self):
        """Return model parameters."""
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Channel dimension of the features returned by ``extract_features``."""
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


# ============================================================================
# Checkpoint handling
# ============================================================================


def _epoch_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(directory: Path, patterns: tuple[str, ...]) -> Path | None:
    """Newest checkpoint (highest epoch number) in ``directory``; ``final.pth`` wins if present."""
    final = directory / "final.pth"
    if final.exists():
        return final
    candidates = [
        p for pattern in patterns for p in directory.glob(pattern) if not p.name.endswith((".optim.pth", ".tmp"))
    ]
    return max(candidates, key=_epoch_number, default=None)


def _resolve_checkpoint(checkpoint_path: str | Path | None, default_dir: str, patterns: tuple[str, ...]) -> Path:
    """Find the checkpoint file; raises FileNotFoundError instead of silently using random weights."""
    roots = [Path.cwd(), Path(__file__).parent.parent.parent]  # as before: also relative to the repo root

    def find(path: Path) -> Path | None:
        for candidate in [path] if path.is_absolute() else [root / path for root in roots]:
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                found = _latest_checkpoint(candidate, patterns)
                if found is not None:
                    return found
        return None

    wanted = Path(checkpoint_path) if checkpoint_path else Path(default_dir)
    found = find(wanted)
    if found is None:
        hint = "" if checkpoint_path else " (pass the path to a checkpoint explicitly, e.g. --teacher-checkpoint)"
        raise FileNotFoundError(f"No teacher checkpoint found at {wanted}{hint}")
    return found


def _load_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # e.g. checkpoints that also pickled non-tensor objects
        logger.warning(f"weights_only loading of {path} failed ({type(exc).__name__}), retrying without it")
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_backbone(ckpt: dict, sections: tuple[str, ...], prefixes: tuple[str, ...], source: Path):
    """Pick the encoder weights out of a checkpoint and strip their prefix.

    Args:
        ckpt: loaded checkpoint (flat state dict or a dict with the state dict in one of ``sections``)
        sections: names of entries that may hold the model state dict, tried in order
        prefixes: key prefixes of the encoder inside that state dict, tried in order
    """
    state = ckpt
    for name in sections:
        if isinstance(ckpt.get(name), dict):
            state = ckpt[name]
            break
    for prefix in prefixes:
        key = f"{prefix}patch_embed.proj.weight"
        if key in state:
            return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    raise ValueError(
        f"{source} has no encoder weights (looked for {[p + 'patch_embed.proj.weight' for p in prefixes]}). "
        "It was probably written by a version of the SSL trainer from before the ViT refactoring; "
        "those weights are not compatible, the SSL model has to be retrained."
    )


# Heads per embedding dim for the published ViT sizes (all other widths: 64 per head).
_HEADS_BY_DIM = {v.embed_dim: v.num_heads for v in ViTVariant}


def _config_from_encoder_state(state: dict, variant: str) -> tuple[ViTConfig, str]:
    """Rebuild the ViTConfig of an encoder from the shapes of its weights.

    Returns the config and the name of the matching ViTVariant ("custom" if none matches).
    """
    proj = state["patch_embed.proj.weight"]  # (embed_dim, in_chans, patch, patch)
    embed_dim, in_chans, patch_size = proj.shape[0], proj.shape[1], proj.shape[2]
    depth = 1 + max(int(m.group(1)) for k in state if (m := re.match(r"blocks\.(\d+)\.", k)))
    has_cls = "cls_token" in state
    pos = state["pos_embed"]  # (1, N (+1), embed_dim), a persistent buffer
    grid = math.isqrt(pos.shape[1] - (1 if has_cls else 0))
    if grid * grid != pos.shape[1] - (1 if has_cls else 0):
        raise ValueError(f"Positional embedding with {pos.shape[1]} tokens is not a square grid")

    mlp_ratio = state["blocks.0.mlp.0.weight"].shape[0] / embed_dim
    num_heads = _HEADS_BY_DIM.get(embed_dim, max(1, embed_dim // 64))

    matching = [v for v in ViTVariant if v.embed_dim == embed_dim and v.depth == depth]
    inferred = matching[0].name if matching else "custom"
    wanted = _VARIANT_ALIASES.get(str(variant).strip().lower())
    if str(variant).strip().lower() != "auto" and wanted is None:
        raise ValueError(f"Unknown teacher variant {variant!r}, use 'auto' or one of {sorted(_VARIANT_ALIASES)}")
    if wanted is not None and wanted != inferred:
        logger.warning(
            f"Requested teacher variant {variant!r} but the checkpoint is {inferred} "
            f"(embed_dim={embed_dim}, depth={depth}); using the checkpoint's architecture"
        )

    config = ViTConfig.from_variant(
        ViTVariant[inferred] if inferred != "custom" else ViTVariant.S,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_size=patch_size,
        base_res=grid * patch_size,
        in_chans=in_chans,
        add_cls_token=has_cls,
        qkv_bias="blocks.0.qkv.bias" in state,
        norm_layer="LayerNorm" if "norm.bias" in state else "RMSNorm",
        drop_path_rate=0.0,
    )
    return config, inferred


# ============================================================================
# Teachers
# ============================================================================


class _VitTeacher(TeacherModel):
    """Shared implementation: a frozen ``VitEncoder`` that turns images into a (B, C, H', W') feature map."""

    NAME = "ViT"
    DEFAULT_DIR = "checkpoints"
    PATTERNS: tuple[str, ...] = ("*.pth",)
    SECTIONS: tuple[str, ...] = ("model_state_dict", "model")
    PREFIXES: tuple[str, ...] = ("",)

    def __init__(
        self,
        checkpoint_path: str | None = None,
        variant: str = "auto",
        input_size: int | None = None,
        input_norm: str = "imagenet",
        use_amp: bool = True,
    ):
        """
        Args:
            checkpoint_path: File (or directory, the newest checkpoint is used) of the SSL model.
                If None, ``DEFAULT_DIR`` is searched. A missing checkpoint raises FileNotFoundError.
            variant: "auto" (from the checkpoint) or e.g. "small" / "base" / "L"; the architecture
                always comes from the checkpoint, a differing variant only triggers a warning.
            input_size: Resolution the images are resized to for the teacher. Default: the
                resolution the teacher was trained at. Larger sizes work (the position embedding
                is interpolated) but cost more and are further from the training distribution.
            input_norm: "imagenet": the images are in [0, 1] and are normalized with the ImageNet
                mean / std before the teacher (as in all SSL trainers); "none": the images already
                are normalized.
            use_amp: bfloat16 autocast on CUDA (the returned features are float32).
        """
        if input_norm not in ("imagenet", "none"):
            raise ValueError("input_norm must be 'imagenet' or 'none'")
        path = _resolve_checkpoint(checkpoint_path, self.DEFAULT_DIR, self.PATTERNS)
        state = _extract_backbone(_load_state(path), self.SECTIONS, self.PREFIXES, path)
        config, inferred = _config_from_encoder_state(state, variant)

        self.encoder = VitEncoder(config=config)
        self.encoder.load_state_dict(state, strict=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.variant = inferred
        self.checkpoint_path = path
        self.config = config
        self._input_size = int(input_size) if input_size else config.base_res
        self._normalize = input_norm == "imagenet"
        self._use_amp = use_amp
        self._device = torch.device("cpu")
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        logger.info(
            f"Loaded {self.NAME} teacher ViT-{inferred}/{config.patch_size} "
            f"(dim={config.embed_dim}, depth={config.depth}, cls={config.add_cls_token}) from {path}"
        )

    def to(self, device) -> "_VitTeacher":
        self._device = torch.device(device)
        self.encoder = self.encoder.to(self._device)
        self._mean = self._mean.to(self._device)
        self._std = self._std.to(self._device)
        return self

    def eval(self) -> None:
        self.encoder.eval()

    def parameters(self):
        return self.encoder.parameters()

    @property
    def feature_dim(self) -> int:
        return self.encoder.embed_dim

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def model_name(self) -> str:
        return f"{self.NAME}-{self.variant}"

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Spatial patch features of the encoder.

        The images are resized to ``input_size`` (a square), optionally ImageNet-normalized, and
        encoded. A CLS token, if the encoder has one (MAE, DINO), is dropped; I-JEPA has none.

        Returns:
            (features (B, C, H', W') float32, C, patch_size)
        """
        size = self._input_size
        if x.shape[2] != size or x.shape[3] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        if self._normalize:
            x = (x - self._mean) / self._std

        patch = self.encoder._patch_size
        use_amp = self._use_amp and x.device.type == "cuda"
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=use_amp):
            tokens = self.encoder.forward_full(x)  # (B, [CLS +] N, C)
        if self.encoder.add_cls_token:
            tokens = tokens[:, 1:]

        batch, num_tokens, channels = tokens.shape
        grid_h, grid_w = x.shape[2] // patch, x.shape[3] // patch
        if num_tokens != grid_h * grid_w:
            raise RuntimeError(f"Teacher produced {num_tokens} tokens for a {grid_h}x{grid_w} patch grid")
        features = tokens.float().reshape(batch, grid_h, grid_w, channels).permute(0, 3, 1, 2).contiguous()
        return features, channels, patch


class MAETeacher(_VitTeacher):
    """
    Masked Autoencoder (MAE) as a teacher model.

    MAE learns to reconstruct masked patches, developing a strong understanding
    of spatial structure and texture. These features are particularly useful
    for object detection tasks.
    """

    NAME = "MAE"
    DEFAULT_DIR = "mae_checkpoints"
    PATTERNS = ("checkpoint_*.pth",)
    SECTIONS = ("model_state_dict", "model")
    PREFIXES = ("encoder.",)


class DINOTeacher(_VitTeacher):
    """
    DINO (Vision Transformer with knowledge distillation) as a teacher model.

    The EMA teacher backbone of the DINO session is used (as in the DINO paper's evaluation).
    """

    NAME = "DINO"
    DEFAULT_DIR = "dino_checkpoints"
    PATTERNS = ("epoch_*.pth",)
    SECTIONS = ("teacher", "model_state_dict")
    PREFIXES = ("backbone.vit.", "vit.", "")


class IJEPATeacher(_VitTeacher):
    """
    I-JEPA (Image Joint-Embedding Predictive Architecture) as a teacher model.

    The EMA target encoder is used (as in the I-JEPA paper's evaluation). It has no CLS token, so all
    tokens are patch tokens.
    """

    NAME = "I-JEPA"
    DEFAULT_DIR = "i_jepa_checkpoints"
    PATTERNS = ("epoch_*.pth",)
    SECTIONS = ("model", "model_state_dict")
    PREFIXES = ("target_encoder.", "context_encoder.")


def create_teacher_model(
    teacher_name: str,
    checkpoint_path: str | None = None,
    variant: str = "auto",
    **kwargs,
) -> TeacherModel | None:
    """
    Factory function to create a teacher model.

    Args:
        teacher_name: Name of the teacher model ('mae', 'dino', 'ijepa', 'none')
        checkpoint_path: Path to a checkpoint file or a checkpoint directory (newest file)
        variant: "auto" (from the checkpoint) or a name like 'small', 'base', 'L' (checked only)
        **kwargs: ``input_size``, ``input_norm``, ``use_amp`` (see ``_VitTeacher``)

    Returns:
        TeacherModel instance, or None for 'none'.

    Raises:
        ValueError: unknown teacher name or incompatible checkpoint
        FileNotFoundError: no checkpoint found. This is deliberately an error: a randomly initialized
            teacher would silently train the detector against noise.
    """
    teacher_name_lower = teacher_name.lower()
    teachers = {"mae": MAETeacher, "dino": DINOTeacher, "ijepa": IJEPATeacher}

    if teacher_name_lower == "none":
        return None
    if teacher_name_lower not in teachers:
        raise ValueError(f"Unknown teacher model: {teacher_name}. Supported: 'mae', 'dino', 'ijepa', 'none'")
    return teachers[teacher_name_lower](checkpoint_path=checkpoint_path, variant=variant, **kwargs)
