import os
from pathlib import Path

import torch
from torch import nn

from components.utils.logger import logger
from components.vit.mae_defs import MAEConfig
from components.vit.position_embedding import build_2d_sincos_position_embedding
from components.vit.vit_block import VitBlock
from components.vit.vit_encoder import VitEncoder


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

    Architecture (sizes come from ``MAEConfig.variant``, T / S / M / B / L / XL):
    - Encoder: a ``VitEncoder`` (same blocks as every other model in this package) with CLS token
    - Decoder: lightweight transformer (default 512-dim, 8 blocks for ViT-B/L/XL)
    - Patch embedding: e.g. 16x16 patches at 224x224 resolution (196 patches)

    Key hyperparameters:
    - mask_ratio: Fraction of patches to mask (typically 0.75 = 75%)
    - encoder_dim: Encoder embedding dimension
    - decoder_dim: Decoder embedding dimension (usually smaller)
    - norm_pix_loss: Reconstruct per-patch normalized pixels (MAE paper; better features)
    - Position embeddings: Sinusoidal (fixed buffers, not parameters)

    The encoder lives in ``self.encoder``; ``patch_embed``, ``cls_token``, ``pos_embed_enc``,
    ``encoder_blocks`` and ``encoder_norm`` are kept as read-only aliases of its parts. Note that the
    state-dict keys changed (``encoder.blocks...``), so checkpoints of the previous version cannot be
    loaded.

    Improvement opportunities:
        - Support variable input resolutions in the decoder
        - Gradient checkpointing for memory efficiency
        - Distributed training with DDP/FSDP
        - Support for different masking strategies (block masking, temporal masking)
        - Contrastive learning combined with reconstruction
    """

    def __init__(self, config: MAEConfig) -> None:
        """
        Initialize MAE model.

        Args:
            config: contains the configs about the model size etc

        Example:
            >>> mae = MAE(mae_config("B"))
            >>> mae.load_checkpoint()  # Load pre-trained weights
        """
        super().__init__()

        self.cfg = config
        self.in_chans = config.in_chans

        if (config.image_size % config.patch_size) != 0:
            raise ValueError(f"img_size {config.image_size} must be divisible by patch_size {config.patch_size}")
        self.grid_size = config.image_size // config.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = config.patch_size * config.patch_size * config.in_chans

        # Encoder: patch embedding, CLS token, fixed sin-cos position embedding, blocks, final norm
        self.encoder = VitEncoder(config=config.vit_config())

        # Build decoder components: embedding, mask token, position embeddings, and transformer blocks
        self.decoder_embed = nn.Linear(config.encoder_dim, config.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
        self.register_buffer(
            "pos_embed_dec",
            build_2d_sincos_position_embedding(self.grid_size, config.decoder_dim),
        )

        # Build decoder blocks: a stack of transformer blocks for reconstructing masked patches
        self.decoder_blocks = nn.ModuleList(
            [
                VitBlock(
                    dim=config.decoder_dim,
                    num_heads=config.decoder_heads,
                    mlp_ratio=config.decoder_mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(config.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(config.decoder_dim)
        self.decoder_pred = nn.Linear(config.decoder_dim, self.patch_dim)

        self._init_weights()

        self.path_final_ckpt = Path("mae_checkpoints") / "final.pth"

    # ---- aliases kept for the old attribute names ----
    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed

    @property
    def cls_token(self) -> nn.Parameter:
        return self.encoder.cls_token

    @property
    def pos_embed_enc(self) -> torch.Tensor:
        return self.encoder.pos_embed

    @property
    def encoder_blocks(self) -> nn.ModuleList:
        return self.encoder.blocks

    @property
    def encoder_norm(self) -> nn.Module:
        return self.encoder.norm

    def _init_weights(self) -> None:
        """
        Initialize model parameters as in the MAE reference implementation.

        Strategy:
        - CLS and MASK tokens: Normal distribution (std=0.02)
        - Position embeddings: Pre-computed sinusoidal (fixed buffers)
        - Linear layers and the patch projection (as a flattened linear layer): Xavier uniform, zero bias
        - Norm layers: weight=1, bias=0 (the defaults)

        Proper initialization is critical for transformer training stability.
        """
        nn.init.normal_(self.encoder.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        proj = self.encoder.patch_embed.proj
        nn.init.xavier_uniform_(proj.weight.view(proj.weight.shape[0], -1))
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches for comparison with predictions.

        Reshapes images from (B, C, H, W) to (B, num_patches, patch_dim)
        where patch_dim = patch_size^2 * C = 768 for 16x16 RGB patches.

        Args:
            imgs: Images of shape (batch, channels, height, width)

        Returns:
            Patches of shape (batch, num_patches, patch_dim)

        Example:
            >>> imgs = torch.randn(2, 3, 224, 224)
            >>> patches = mae.patchify(imgs)  # (2, 196, 768)
        """
        p = self.cfg.patch_size
        h, w = imgs.shape[2] // p, imgs.shape[3] // p
        x = imgs.reshape(imgs.shape[0], self.in_chans, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)  # Rearrange to (batch, h, w, p, p, channels)
        return x.reshape(imgs.shape[0], h * w, p * p * self.in_chans)

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
        x = x.reshape(x.shape[0], n, n, p, p, self.in_chans)
        x = torch.einsum("nhwpqc->nchpwq", x)  # Rearrange to (batch, channels, n, p, n, p)
        return x.reshape(x.shape[0], self.in_chans, n * p, n * p)

    @staticmethod
    def _masking_indices(
        n: int,
        l: int,
        mask_ratio: float,
        device: torch.device,  # noqa: E741
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random per-sample split into kept and masked patches.

        Returns:
            ids_keep (n, len_keep), mask (n, l) float32 with 1 = masked, ids_restore (n, l)
        """
        # the epsilon guards against float error, e.g. 100 * (1 - 0.7) = 30.000000000000004
        len_keep = max(1, int(l * (1 - mask_ratio) + 1e-6))

        noise = torch.rand(n, l, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        # position i is masked iff its rank in the shuffle is >= len_keep
        mask = (ids_restore >= len_keep).to(torch.float32)
        return ids_keep, mask, ids_restore

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        ids_keep, mask, ids_restore = self._masking_indices(n, l, mask_ratio, x.device)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))
        return x_masked, mask, ids_restore

    def forward_encoder_full(self, imgs: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        """
        Encode all patches through the transformer encoder without masking.

        This helper is used by the YOLO distillation path to obtain deterministic
        teacher features from the MAE encoder. It mirrors the visible-patch encoder
        logic but skips masking unless a positive mask ratio is requested.

        Returns:
            Tokens (batch, 1 + num_patches, encoder_dim), CLS token first. Other input resolutions
            than ``image_size`` work here (the position embedding is interpolated).
        """
        x, _, _ = self.forward_encoder(imgs, mask_ratio)
        return x

    def forward_encoder(
        self, imgs: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Encode visible patches through transformer encoder.

        Process:
        1. Convert images to patch embeddings and add positional embeddings
        2. Randomly mask patches (skipped for mask_ratio = 0)
        3. Prepend CLS token
        4. Process through transformer encoder blocks
        5. Apply layer normalization

        Args:
            imgs: Input images (batch, C, H, W)
            mask_ratio: Fraction of patches to mask

        Returns:
            Tuple of:
            - x: Encoded features (batch, num_keep+1, encoder_dim)
            - mask: Binary mask showing which patches were masked (None without masking)
            - ids_restore: Indices to restore original patch order (None without masking)
        """
        if mask_ratio > 0.0:
            p = self.cfg.patch_size
            n_patches = (imgs.shape[2] // p) * (imgs.shape[3] // p)
            ids_keep, mask, ids_restore = self._masking_indices(imgs.shape[0], n_patches, mask_ratio, imgs.device)
            x = self.encoder.forward_visible(imgs, ids_keep)
            return x, mask, ids_restore

        # For full encoder: no masking
        return self.encoder.forward_full(imgs), None, None

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
        if ids_restore.shape[1] != self.num_patches:
            raise ValueError(
                f"The decoder expects {self.num_patches} patches ({self.cfg.image_size}px images), "
                f"got {ids_restore.shape[1]}"
            )

        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.to(x.dtype).expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.pos_embed_dec

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        return self.decoder_pred(x[:, 1:, :])

    def _pixel_stats(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        return mean, var

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MAE reconstruction loss.

        Key insight: Only compute loss on masked patches. This forces meaningful
        reconstruction of difficult regions while encoder focuses on visible patches.

        Loss = MSE(predicted_patches, target_patches) masked by mask, averaged over masked patches.
        With ``norm_pix_loss`` the target patches are normalized by their own mean and variance
        first (He et al., 2021), so ``pred`` lives in that normalized space; use
        ``denormalize_pred`` to turn it back into pixels. The loss is computed in float32.

        Args:
            imgs: Original images (batch, C, H, W)
            pred: Predicted patches (batch, num_patches, patch_dim)
            mask: Binary mask (batch, num_patches) where 1=masked

        Returns:
            Tuple of:
            - loss: Scalar loss value
            - target: Target patch values (normalized if ``norm_pix_loss``), for visualization

        Note:
            Mean Squared Error (MSE) is used here. Alternatives:
            - L1 loss (more robust to outliers)
            - Smooth L1 loss (hybrid of L1 and L2)
            - Perceptual loss (using pre-trained discriminator)
        """
        target = self.patchify(imgs).float()
        if self.cfg.norm_pix_loss:
            mean, var = self._pixel_stats(target)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred.float() - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, target

    def denormalize_pred(self, imgs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Map ``pred`` back to pixel space (a no-op without ``norm_pix_loss``).

        With ``norm_pix_loss`` the per-patch mean / std of the *original* patches are re-applied,
        as in the reference visualization. Use ``unpatchify`` on the result to get an image.
        """
        if not self.cfg.norm_pix_loss:
            return pred
        mean, var = self._pixel_stats(self.patchify(imgs).float())
        return pred.float() * (var + 1e-6) ** 0.5 + mean

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete forward pass: encode, decode, compute loss.

        Args:
            imgs: Input images (batch, C, H, W)

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

    def save_checkpoint(self, path: Path | None = None) -> None:
        """
        Save model checkpoint to disk.

        Args:
            path: Path to save checkpoint. If None, uses default path from config.

        Note:
            Model is moved to CPU before saving to avoid device-specific issues. The file is
            written atomically, so an interrupted run cannot leave a truncated checkpoint.
        """
        path_ckpt = Path(self.path_final_ckpt if path is None else path)
        path_ckpt.parent.mkdir(parents=True, exist_ok=True)

        state_dict_cpu = {k: v.detach().cpu() for k, v in self.state_dict().items()}

        tmp = path_ckpt.with_name(path_ckpt.name + ".tmp")
        torch.save(state_dict_cpu, tmp)
        os.replace(tmp, path_ckpt)

    def load_checkpoint(self, path: Path | None = None, device: str | None = None) -> None:
        """
        Load model checkpoint from disk.

        Args:
            path: Path to checkpoint file. If None, uses default path from config.
            device: Device to load checkpoint to ('cpu', 'cuda', 'cuda:0', etc.).
                   If None, uses current device.

        Note:
            Prints warning if checkpoint not found but doesn't raise exception.
        """
        path_ckpt = Path(self.path_final_ckpt if path is None else path)
        logger.info(f"trying to load ckpt {path_ckpt}")

        if not path_ckpt.exists():
            logger.warning(f"{path_ckpt} not found, cannot load")
            return

        state_dict = torch.load(path_ckpt, map_location=device, weights_only=True)
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        self.load_state_dict(state_dict)
