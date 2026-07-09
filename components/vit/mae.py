from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from components.utils.logger import logger
from components.vit.patch_embed import PatchEmbed
from components.vit.position_embedding import build_2d_sincos_position_embedding
from components.vit.transformer_block import TransformerBlock


@dataclass(frozen=True)
class MAEVariantConfig:
    """
    Configuration for different MAE model variants.

    Attributes:
        dataset_size: Total number of samples in the dataset
        image_size: Input image resolution (pixels)
        patch_size: Size of image patches (pixels)
        batch_size: Training batch size
        encoder_dim: Embedding dimension for transformer encoder
        encoder_depth: Number of transformer blocks in encoder
        encoder_heads: Number of attention heads in encoder
        decoder_dim: Embedding dimension for transformer decoder
        decoder_depth: Number of transformer blocks in decoder
        decoder_heads: Number of attention heads in decoder
        mask_ratio: Fraction of patches to mask during training (0-1)
        learning_rate: Initial learning rate for optimization
    """

    dataset_size: int
    image_size: int
    patch_size: int
    batch_size: int
    encoder_dim: int
    encoder_depth: int
    encoder_heads: int
    decoder_dim: int
    decoder_depth: int
    decoder_heads: int
    mask_ratio: float
    learning_rate: float


VARIANT_CONFIG = {
    # Fits typical 4GB GPUs for debugging (with AMP).
    "cifar10": MAEVariantConfig(
        dataset_size=50_000,
        image_size=32,
        patch_size=4,
        batch_size=32 * 1,
        encoder_dim=384,
        encoder_depth=8,
        encoder_heads=6,
        decoder_dim=192,
        decoder_depth=4,
        decoder_heads=6,
        mask_ratio=0.75,
        learning_rate=1e-3,
    ),
    # MAE-Base style scale for larger GPUs (32-40GB+ suggested).
    "imagenet": MAEVariantConfig(
        dataset_size=1_281_167,
        image_size=256,  # ↑ use full 256 for better spatial detail
        patch_size=16,
        batch_size=2 * 1,
        # ---------------- Encoder ----------------
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        # ---------------- Decoder ----------------
        decoder_dim=768,  # ↑ stronger decoder for sharper teacher features
        decoder_depth=8,  # keep 8 (good balance)
        decoder_heads=12,  # ↓ from 16 → 12 (matches encoder, more stable)
        # ---------------- Masking ----------------
        mask_ratio=0.6,  # ↓ from 0.75 → 0.6 (better for dense tasks)
        # ---------------- Optimization ----------------
        learning_rate=1.5e-4,  # unchanged; still optimal for MAE-B
        # missing fields you should add:
        # use_sin_pos_embed=True,   # recommended: 2D sin-cos positional embeddings
        # drop_path_rate=0.1,       # stable for ViT-B encoder
        # decoder_drop_path_rate=0.0,
        # loss_type="mse",          # better for YOLO distillation
    ),
    # Same MAE scale as ImageNet but intended for smaller ImageNet-style subsets.
    "imagenet_mini": MAEVariantConfig(
        dataset_size=100_000,
        image_size=224,
        patch_size=16,
        batch_size=32 * 1,
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_heads=16,
        mask_ratio=0.75,
        learning_rate=1.5e-4,
    ),
    "coco": MAEVariantConfig(
        dataset_size=1_281_167,
        image_size=256,
        patch_size=16,
        batch_size=32 * 1,
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_heads=16,
        mask_ratio=0.75,
        learning_rate=1.5e-4,
    ),
}


class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) for self-supervised visual representation learning.

    MAE learns representations by:
    1. Randomly masking 75% of image patches
    2. Encoding only the visible patches with a transformer
    3. Decoding all patches (including masked) to reconstruct the image
    4. Computing reconstruction loss only on masked patches

    This asymmetric design (small encoder, large decoder) learns strong representations
    while being computationally efficient. Pre-training on unlabeled data enables better
    fine-tuning on downstream tasks like object detection (YOLO).

    Architecture:
    - Encoder: Vision Transformer (12 blocks, 768-dim for ImageNet)
    - Decoder: Vision Transformer (8 blocks, 512-dim for ImageNet)
    - Patch embedding: 16x16 patches at 224x224 resolution (196 patches)

    Key hyperparameters:
    - mask_ratio: Fraction of patches to mask (typically 0.75 = 75%)
    - encoder_dim: Encoder embedding dimension
    - decoder_dim: Decoder embedding dimension (usually smaller)
    - Position embeddings: Sinusoidal (non-learnable, freezable)

    Improvement opportunities:
        - Support variable input resolutions
        - Gradient checkpointing for memory efficiency
        - Distributed training with DDP/FSDP
        - Support for different masking strategies (block masking, temporal masking)
        - Exponential Moving Average (EMA) updates for stability
        - Contrastive learning combined with reconstruction
    """

    def __init__(self, variant: str, in_chans: int = 3) -> None:
        """
        Initialize MAE model with variant-specific configuration.

        Args:
            variant: Model variant ('cifar10', 'imagenet', or 'coco')
            in_chans: Number of input channels (3 for RGB images)

        Raises:
            KeyError: If variant not in VARIANT_CONFIG

        Example:
            >>> mae = MAE('imagenet')
            >>> mae.load_checkpoint()  # Load pre-trained weights
        """
        super().__init__()

        self.cfg = VARIANT_CONFIG[variant]
        self.in_chans = in_chans

        # Patch embedding layer: converts images to patch embeddings
        if (self.cfg.image_size % self.cfg.patch_size) != 0:
            raise ValueError(f"img_size {self.cfg.image_size} must be divisible by patch_size {self.cfg.patch_size}")
        self.patch_embed = PatchEmbed(
            patch_size=self.cfg.patch_size,
            in_chans=in_chans,
            embed_dim=self.cfg.encoder_dim,
        )
        self.grid_size = self.cfg.image_size // self.cfg.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = self.cfg.patch_size * self.cfg.patch_size * in_chans

        # Initialize learnable tokens and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.encoder_dim))
        self.pos_embed_enc = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.cfg.encoder_dim), requires_grad=False
        )

        # Build encoder blocks: a stack of transformer blocks for encoding visible patches
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=self.cfg.encoder_dim, num_heads=self.cfg.encoder_heads)
                for _ in range(self.cfg.encoder_depth)
            ]
        )
        # Layer normalization after encoder blocks for stability
        self.encoder_norm = nn.LayerNorm(self.cfg.encoder_dim)

        # Build decoder components: embedding, mask token, position embeddings, and transformer blocks
        self.decoder_embed = nn.Linear(self.cfg.encoder_dim, self.cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.cfg.decoder_dim))
        self.pos_embed_dec = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.cfg.decoder_dim), requires_grad=False
        )

        # Build decoder blocks: a stack of transformer blocks for reconstructing masked patches
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=self.cfg.decoder_dim, num_heads=self.cfg.decoder_heads)
                for _ in range(self.cfg.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.cfg.decoder_dim)
        self.decoder_pred = nn.Linear(self.cfg.decoder_dim, self.patch_dim)

        self._init_weights()

        self.path_final_ckpt = Path(__file__).resolve().parent / "mae_checkpoints" / variant / "final.pth"
        if not self.path_final_ckpt.parent.exists():
            self.path_final_ckpt.parent.mkdir(parents=True, exist_ok=True)

    def _init_weights(self) -> None:
        """
        Initialize model parameters using proper initialization schemes.

        Strategy:
        - CLS and MASK tokens: Normal distribution (std=0.02)
        - Position embeddings: Pre-computed sinusoidal (non-learnable)
        - Linear layers: Xavier uniform initialization
        - LayerNorm: Constant initialization (bias=0, weight=1)

        Proper initialization is critical for transformer training stability.
        """
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        with torch.no_grad():
            self.pos_embed_enc.copy_(build_2d_sincos_position_embedding(self.grid_size, self.cfg.encoder_dim))
            self.pos_embed_dec.copy_(build_2d_sincos_position_embedding(self.grid_size, self.cfg.decoder_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches for comparison with predictions.

        Reshapes images from (B, C, H, W) to (B, num_patches, patch_dim)
        where patch_dim = patch_size^2 * C = 48 for 16x16 RGB patches.

        Args:
            imgs: Images of shape (batch, channels, height, width)

        Returns:
            Patches of shape (batch, num_patches, patch_dim)

        Example:
            >>> imgs = torch.randn(2, 3, 224, 224)
            >>> patches = mae.patchify(imgs)  # (2, 196, 768)
        """
        p = self.cfg.patch_size
        n = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, n, p, n, p))
        x = torch.einsum("nchpwq->nhwpqc", x)  # Rearrange to (batch, n, n, p, p, channels)
        return x.reshape(shape=(imgs.shape[0], n * n, p * p * self.in_chans))

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images for visualization.

        Reverses the patchify operation: (B, num_patches, patch_dim) -> (B, C, H, W)

        Args:
            x: Patches of shape (batch, num_patches, patch_dim)

        Returns:
            Images of shape (batch, channels, height, width)

        Note:
            num_patches must be a perfect square: sqrt(num_patches) = height/patch_size
        """
        p = self.cfg.patch_size
        n = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], n, n, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)  # Rearrange to (batch, channels, n, p, n, p)
        return x.reshape(shape=(x.shape[0], self.in_chans, n * p, n * p))

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches and keep only visible patches.

        The core of MAE: randomly select patches to mask, encode only visible patches.
        This forces the model to learn meaningful representations by reconstruction.

        Algorithm:
        1. Generate random noise for each patch
        2. Sort by noise to determine mask/keep split
        3. Keep patches with lowest noise (by design, random)
        4. Create restoration indices to unpermute patches

        Args:
            x: Patch embeddings of shape (batch, num_patches, dim)
            mask_ratio: Fraction of patches to mask (e.g., 0.75 = mask 75%)

        Returns:
            Tuple of:
            - x_masked: Visible patches only (batch, num_keep, dim)
            - mask: Binary mask (batch, num_patches) where 1=masked, 0=visible
            - ids_restore: Indices to restore original patch order

        Example:
            >>> x = torch.randn(2, 196, 768)
            >>> x_masked, mask, restore_ids = mae.random_masking(x, mask_ratio=0.75)
            >>> x_masked.shape  # (2, 49, 768) - only 25% of patches
        """
        n, l, d = x.shape  # noqa: E741
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder_full(self, imgs: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        """
        Encode all patches through the transformer encoder without masking.

        This helper is used by the YOLO distillation path to obtain deterministic
        teacher features from the MAE encoder. It mirrors the visible-patch encoder
        logic but skips masking unless a positive mask ratio is requested.
        """
        x, _, _ = self.forward_encoder(imgs, mask_ratio)
        return x

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode visible patches through transformer encoder.

        Process:
        1. Convert images to patch embeddings
        2. Add positional embeddings
        3. Randomly mask patches
        4. Prepend CLS token
        5. Process through transformer encoder blocks
        6. Apply layer normalization

        Args:
            imgs: Input images (batch, 3, H, W)
            mask_ratio: Fraction of patches to mask

        Returns:
            Tuple of:
            - x: Encoded features (batch, num_keep+1, encoder_dim)
            - mask: Binary mask showing which patches were masked
            - ids_restore: Indices to restore original patch order
        """
        # 1. Patch embedding
        x = self.patch_embed(imgs)
        # 2. Positional embedding
        x = x + self.pos_embed_enc[:, 1:, :]

        # 3. Optional masking
        if mask_ratio > 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            # For full encoder: no masking
            mask = None
            ids_restore = None

        # 4. CLS token
        cls_token = self.cls_token + self.pos_embed_enc[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 5. Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)

        # 6. Final norm
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Decode encoder features and reconstruct all patches.

        Process:
        1. Project encoder embeddings to decoder dimension
        2. Create mask tokens for masked positions
        3. Concatenate visible and mask tokens in original order
        4. Add positional embeddings
        5. Process through transformer decoder blocks
        6. Predict patch values

        Args:
            x: Encoder output (batch, num_keep+1, encoder_dim)
            ids_restore: Indices to restore original patch order

        Returns:
            Predicted patches (batch, num_patches, patch_dim)

        Note:
            The decoder is asymmetrically designed to be smaller than encoder,
            which reduces computational cost while maintaining quality.
        """
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.pos_embed_dec

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        return self.decoder_pred(x[:, 1:, :])

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MAE reconstruction loss.

        Key insight: Only compute loss on masked patches. This forces meaningful
        reconstruction of difficult regions while encoder focuses on visible patches.

        Loss = MSE(predicted_patches, target_patches) masked by mask, averaged over masked patches.

        Args:
            imgs: Original images (batch, 3, H, W)
            pred: Predicted patches (batch, num_patches, patch_dim)
            mask: Binary mask (batch, num_patches) where 1=masked

        Returns:
            Tuple of:
            - loss: Scalar loss value
            - target: Original patch values for visualization

        Note:
            Mean Squared Error (MSE) is used here. Alternatives:
            - L1 loss (more robust to outliers)
            - Smooth L1 loss (hybrid of L1 and L2)
            - Perceptual loss (using pre-trained discriminator)
        """
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, target

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete forward pass: encode, decode, compute loss.

        Args:
            imgs: Input images (batch, 3, H, W)

        Returns:
            Tuple of:
            - loss: Reconstruction loss on masked patches
            - pred: Predicted patches (batch, num_patches, patch_dim)
            - target: Target patches (batch, num_patches, patch_dim)
            - mask: Binary mask (batch, num_patches)
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, self.cfg.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss, target = self.forward_loss(imgs, pred, mask)
        return loss, pred, target, mask

    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        """
        Save model checkpoint to disk.

        Args:
            path: Path to save checkpoint. If None, uses default path from config.

        Note:
            Model is moved to CPU before saving to avoid device-specific issues.
        """
        path_ckpt = self.path_final_ckpt if path is None else path

        state_dict_cpu = {k: v.cpu() for k, v in self.state_dict().items()}

        torch.save(state_dict_cpu, path_ckpt)

    def load_checkpoint(self, path: Optional[Path] = None, device: Optional[str] = None) -> None:
        """
        Load model checkpoint from disk.

        Args:
            path: Path to checkpoint file. If None, uses default path from config.
            device: Device to load checkpoint to ('cpu', 'cuda', 'cuda:0', etc.).
                   If None, uses current device.

        Note:
            Prints warning if checkpoint not found but doesn't raise exception.
        """
        path_ckpt = self.path_final_ckpt if path is None else path

        if not path_ckpt.exists():
            logger.info(f"{path_ckpt} not found, cannot load")
            return

        state_dict = torch.load(path_ckpt, map_location=device)
        self.load_state_dict(state_dict)
