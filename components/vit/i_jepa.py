from __future__ import annotations

import copy

import torch
from torch import nn

from components.vit.i_jepa_defs import IJEPAConfig
from components.vit.i_jepa_masks import sample_context_and_target_masks, sample_rect_masks
from components.vit.vit_block import VitBlock
from components.vit.vit_encoder import VitEncoder


class I_JEPA(nn.Module):
    """
    I-JEPA model (architecture only).

    This module includes:
    - Context encoder (trainable)
    - Target encoder (EMA teacher, frozen)
    - Latent-space predictor for target blocks

    No training session/loss orchestration is included here.
    """

    def __init__(self, config: IJEPAConfig) -> None:
        super().__init__()
        self.cfg = config

        # Backbone: any ViTVariant (config.variant), patch tokens only (no CLS token).
        self.context_encoder = VitEncoder(config=config.vit_config())
        # The target encoder starts as an exact copy of the context encoder and is then updated by EMA.
        self.target_encoder = copy.deepcopy(self.context_encoder)

        # Predictor: context latent -> target latent
        self.pred_mask_token = nn.Parameter(torch.zeros(1, 1, config.predictor_dim))
        self.pred_in = nn.Linear(config.embed_dim, config.predictor_dim)
        self.pred_in_pos = nn.Linear(config.embed_dim, config.predictor_dim)
        self.pred_blocks = nn.ModuleList(
            [
                VitBlock(
                    dim=config.predictor_dim,
                    num_heads=config.predictor_heads,
                    mlp_ratio=config.predictor_mlp_ratio,
                    attn_drop=config.attn_drop,
                    proj_drop=config.proj_drop,
                    drop_path=0.0,
                    qkv_bias=True,
                )
                for _ in range(config.predictor_depth)
            ]
        )
        self.pred_norm = nn.LayerNorm(config.predictor_dim)
        self.pred_out = nn.Linear(config.predictor_dim, config.embed_dim)

        self.grid_size = self.cfg.image_size // self.cfg.patch_size
        self.num_patches = self.grid_size * self.grid_size

        self._freeze_target_encoder()
        self._init_predictor()

    def train(self, mode: bool = True) -> I_JEPA:
        """Switch train/eval mode; the EMA target encoder always stays in eval mode."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _init_predictor(self) -> None:
        nn.init.normal_(self.pred_mask_token, std=0.02)
        for m in [self.pred_in, self.pred_in_pos, self.pred_out, self.pred_norm, *self.pred_blocks.modules()]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _init_target_encoder_from_context(self) -> None:
        self.target_encoder.load_state_dict(self.context_encoder.state_dict(), strict=True)

    def _freeze_target_encoder(self) -> None:
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

    @torch.no_grad()
    def momentum_update_target_encoder(self, momentum: float) -> None:
        """EMA update for target encoder parameters."""
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("momentum must be in [0, 1]")

        # fused multi-tensor update instead of two kernels per parameter
        target_params = list(self.target_encoder.parameters())
        context_params = [p.detach() for p in self.context_encoder.parameters()]
        torch._foreach_mul_(target_params, momentum)
        torch._foreach_add_(target_params, context_params, alpha=1.0 - momentum)

    def _sample_rect_mask(
        self,
        grid_h: int,
        grid_w: int,
        scale_min: float,
        scale_max: float,
        ar_min: float,
        ar_max: float,
        device: torch.device,
        num_masks: int = 1,
    ) -> torch.Tensor:
        """Sample rectangular boolean masks on a patch grid (see ``i_jepa_masks.sample_rect_masks``).

        Returns:
            Tensor of shape (num_masks, grid_h * grid_w) with bool dtype, or (grid_h * grid_w,)
            for num_masks == 1 (backward compatible).
        """
        masks = sample_rect_masks(grid_h, grid_w, scale_min, scale_max, ar_min, ar_max, num_masks, device)
        return masks if num_masks > 1 else masks[0]

    def _sample_context_and_target_masks(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Create per-image context masks and multiple per-image target masks
        (see ``i_jepa_masks.sample_context_and_target_masks``).

        Returns:
            context_keep_mask: (B, N) bool
            target_masks: list[(B, N)] bool
        """
        return sample_context_and_target_masks(self.cfg, batch_size, device)

    @staticmethod
    def _target_block_indices(target_masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Token indices of all target blocks, padded to one common length.

        The sort is stable, so every block lists its tokens in ascending patch order; computing this
        once guarantees that predictions, target latents and valid masks refer to the same patches.

        Returns:
            indices: (K, B, M) long, patch index of every (padded) target token
            valid: (K, B, M) bool, False for padding
        """
        masks = torch.stack(target_masks)  # (K, B, N)
        num_blocks, bsz, num_patches = masks.shape
        flat = masks.reshape(num_blocks * bsz, num_patches)

        counts = flat.sum(dim=1)  # (K*B,)
        max_count = max(int(counts.max().item()), 1)  # the only host sync

        order = torch.sort(flat.to(torch.uint8), dim=1, descending=True, stable=True)[1][:, :max_count]
        valid = torch.arange(max_count, device=flat.device).unsqueeze(0) < counts.unsqueeze(1)
        return order.view(num_blocks, bsz, max_count), valid.view(num_blocks, bsz, max_count)

    def _gather_padded_target_tokens(
        self,
        full_tokens: torch.Tensor,
        target_masks: list[torch.Tensor],
        block_index: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Gather per-block target tokens and pad all blocks to a common length.

        Args:
            block_index: Optional result of ``_target_block_indices`` (computed if omitted).

        Returns:
            target_tokens: list[(B, max_Nt, D)]
            target_valid_masks: list[(B, max_Nt)] bool
        """
        indices, valid = block_index if block_index is not None else self._target_block_indices(target_masks)
        num_blocks, bsz, max_count = indices.shape
        dim = full_tokens.shape[-1]

        flat_idx = indices.permute(1, 0, 2).reshape(bsz, num_blocks * max_count)  # (B, K*M), block-major
        tok = torch.gather(full_tokens, 1, flat_idx.unsqueeze(-1).expand(-1, -1, dim))
        tok = tok.view(bsz, num_blocks, max_count, dim)
        return list(tok.unbind(1)), list(valid.unbind(0))

    def _predict_targets(
        self,
        context_tokens: torch.Tensor,
        context_padding_mask: torch.Tensor | None,
        target_masks: list[torch.Tensor],
        pos_spatial: torch.Tensor,
        block_index: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Predict target-token latents from context tokens and target positions.

        All target blocks are processed in ONE forward pass through the predictor: their queries are
        concatenated after the context tokens. This is valid because target blocks are independent
        (queries only attend to context tokens and to each other, never to padding).
        """
        indices, valid = block_index if block_index is not None else self._target_block_indices(target_masks)
        num_blocks, bsz, max_count = indices.shape
        device = context_tokens.device

        context_z = self.pred_in(context_tokens)  # (B, Nc, D_pred)
        pos_proj = self.pred_in_pos(pos_spatial)  # (1, N, D_pred)
        dim_pred = pos_proj.shape[-1]

        # Target queries: mask token + position embedding of the target patch
        flat_idx = indices.permute(1, 0, 2).reshape(bsz, num_blocks * max_count)  # (B, K*M), block-major
        target_pos = torch.gather(pos_proj.expand(bsz, -1, -1), 1, flat_idx.unsqueeze(-1).expand(-1, -1, dim_pred))
        target_queries = self.pred_mask_token + target_pos  # (B, K*M, D_pred)
        flat_valid = valid.permute(1, 0, 2).reshape(bsz, num_blocks * max_count)

        z = torch.cat([context_z, target_queries], dim=1)  # (B, Nc + K*M, D_pred)

        # Combined padding mask (True = invalid/padding)
        if context_padding_mask is None:
            context_padding_mask = torch.zeros(context_z.shape[:2], dtype=torch.bool, device=device)
        combined_padding = torch.cat([context_padding_mask, ~flat_valid], dim=1)

        for blk in self.pred_blocks:
            z = blk(z, padding_mask=combined_padding)

        z = self.pred_norm(z)
        pred = self.pred_out(z[:, context_z.shape[1] :, :])  # (B, K*M, D_embed)
        pred = pred.view(bsz, num_blocks, max_count, -1)

        return list(pred.unbind(1)), list(valid.unbind(0))

    def forward(
        self,
        imgs: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_masks: list[torch.Tensor] | None = None,
    ) -> dict[str, object]:
        """
        Forward pass for I-JEPA architecture.

        Args:
            imgs: Input images, shape (B, 3, H, W)
            context_mask: Optional pre-computed context mask (B, N). If None, samples masks.
            target_masks: Optional pre-computed target masks list[(B, N)]. If None, samples masks.

        Returns model outputs required for downstream training code:
        - predicted_target_tokens: list[(B, Nt, D)] (one entry per target block, padded to a common Nt)
        - target_tokens: list[(B, Nt, D)]
        - context_tokens: (B, Nc, D)
        - context_mask: (B, N)
        - target_masks: list[(B, N)]
        - target_valid_masks / predicted_target_masks: list[(B, Nt)] bool, False for padding
        """
        if imgs.ndim != 4:
            raise ValueError("imgs must have shape (B, C, H, W)")
        if imgs.shape[2] != self.cfg.image_size or imgs.shape[3] != self.cfg.image_size:
            raise ValueError(f"imgs must be {self.cfg.image_size}x{self.cfg.image_size}, got {tuple(imgs.shape[2:])}")

        bsz = imgs.shape[0]
        device = imgs.device

        # Use provided masks or sample them on-the-fly
        if context_mask is None or target_masks is None:
            context_mask, target_masks = self._sample_context_and_target_masks(batch_size=bsz, device=device)
        else:
            # Validate pre-computed masks
            expected = (bsz, self.num_patches)
            if context_mask.shape != expected:
                raise ValueError(f"context_mask shape {tuple(context_mask.shape)} != expected {expected}")
            if len(target_masks) != self.cfg.num_target_blocks:
                raise ValueError(f"num target_masks {len(target_masks)} != config {self.cfg.num_target_blocks}")
            for i, mask in enumerate(target_masks):
                if mask.shape != expected:
                    raise ValueError(f"target_masks[{i}] shape {tuple(mask.shape)} != expected {expected}")
            context_mask = context_mask.to(torch.bool)
            target_masks = [m.to(torch.bool) for m in target_masks]

        context_tokens, context_padding_mask = self.context_encoder.forward_full(
            imgs,
            patch_keep_mask=context_mask,
            return_padding_mask=True,
        )

        # (the target encoder is kept in eval mode by ``train()``)
        with torch.no_grad():
            full_target_tokens = self.target_encoder.forward_full(
                imgs,
                patch_keep_mask=None,
            )

        h_patch = imgs.shape[2] // self.cfg.patch_size
        w_patch = imgs.shape[3] // self.cfg.patch_size
        pos_all = self.context_encoder.interpolate_pos_encoding(h_patch, w_patch)

        # Sort once; predictions, target latents and valid masks all use the same indices.
        block_index = self._target_block_indices(target_masks)

        predicted_target_tokens, predicted_target_masks = self._predict_targets(
            context_tokens, context_padding_mask, target_masks, pos_all, block_index
        )

        target_tokens, target_valid_masks = self._gather_padded_target_tokens(
            full_target_tokens, target_masks, block_index
        )

        return {
            "predicted_target_tokens": predicted_target_tokens,
            "target_tokens": target_tokens,
            "context_tokens": context_tokens,
            "context_mask": context_mask,
            "target_masks": target_masks,
            "target_valid_masks": target_valid_masks,
            "predicted_target_masks": predicted_target_masks,
        }


def _smoke_test():
    cfg = IJEPAConfig(image_size=64, patch_size=8, variant="T")
    model = I_JEPA(cfg)
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    out = model(x)
    assert "predicted_target_tokens" in out
    assert "target_tokens" in out
    for pred, tgt, valid in zip(out["predicted_target_tokens"], out["target_tokens"], out["target_valid_masks"]):
        assert pred.shape == tgt.shape and pred.shape[:2] == valid.shape


if __name__ == "__main__":
    _smoke_test()
