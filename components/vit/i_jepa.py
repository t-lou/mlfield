from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from components.vit.i_jepa_defs import IJEPAConfig
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

        self.context_encoder = VitEncoder(
            base_res=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            drop_path_rate=config.drop_path_rate,
            qkv_bias=config.qkv_bias,
        )
        self.target_encoder = VitEncoder(
            base_res=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
            drop_path_rate=config.drop_path_rate,
            qkv_bias=config.qkv_bias,
        )
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

        self._init_target_encoder_from_context()
        self._freeze_target_encoder()
        self._init_predictor()

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

        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_c.data, alpha=1.0 - momentum)

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
        """Sample rectangular boolean masks on a patch grid.

        Fully vectorized: no .item() calls, pure GPU tensor operations.

        Args:
            num_masks: Number of masks to sample (default 1 for backward compat)

        Returns:
            Tensor of shape (num_masks, grid_h * grid_w) with bool dtype
        """
        total = grid_h * grid_w

        # Batch sample areas and aspects on GPU (vectorized, no .item())
        areas = torch.empty(num_masks, device=device).uniform_(scale_min, scale_max) * total
        aspects = torch.empty(num_masks, device=device).uniform_(ar_min, ar_max)

        # Compute heights and widths vectorized
        h_vals = torch.round((areas * aspects) ** 0.5).int()
        w_vals = torch.round((areas / aspects) ** 0.5).int()
        h_vals = torch.clamp(h_vals, min=1, max=grid_h)
        w_vals = torch.clamp(w_vals, min=1, max=grid_w)

        # Sample positions vectorized
        top_vals = torch.randint(0, max(1, grid_h - 1), (num_masks,), device=device)
        left_vals = torch.randint(0, max(1, grid_w - 1), (num_masks,), device=device)

        # Clamp heights/widths to fit within grid from sampled positions
        h_vals = torch.clamp(h_vals, max=grid_h - top_vals)
        w_vals = torch.clamp(w_vals, max=grid_w - left_vals)

        # Build all masks at once using broadcasting (NO PYTHON LOOP)
        # Create coordinate grids: (grid_h, grid_w)
        y_grid = torch.arange(grid_h, device=device).view(-1, 1)  # (H, 1)
        x_grid = torch.arange(grid_w, device=device).view(1, -1)  # (1, W)

        # Reshape for broadcasting with batch dimension
        # top_vals: (B, 1, 1), h_vals: (B, 1, 1)
        top_vals_expanded = top_vals.view(-1, 1, 1)  # (B, 1, 1)
        left_vals_expanded = left_vals.view(-1, 1, 1)  # (B, 1, 1)
        h_vals_expanded = h_vals.view(-1, 1, 1)  # (B, 1, 1)
        w_vals_expanded = w_vals.view(-1, 1, 1)  # (B, 1, 1)

        y_grid_expanded = y_grid.view(1, -1, 1)  # (1, H, 1)
        x_grid_expanded = x_grid.view(1, 1, -1)  # (1, 1, W)

        # Create masks via broadcasting (fully vectorized)
        in_y_range = (y_grid_expanded >= top_vals_expanded) & (
            y_grid_expanded < top_vals_expanded + h_vals_expanded
        )  # (B, H, 1)
        in_x_range = (x_grid_expanded >= left_vals_expanded) & (
            x_grid_expanded < left_vals_expanded + w_vals_expanded
        )  # (1, 1, W)

        masks = (in_y_range & in_x_range).view(num_masks, grid_h * grid_w)  # (B, H*W)

        return masks if num_masks > 1 else masks[0]

    def _sample_context_and_target_masks(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Create per-image context masks and multiple per-image target masks.

        Fully vectorized with no .item() calls or Python loops over batch.

        Returns:
            context_keep_mask: (B, N) bool
            target_masks: list[(B, N) bool]
        """
        grid_h = self.grid_size
        grid_w = self.grid_size
        grid_n = grid_h * grid_w

        # Batch sample all context masks at once
        context_masks_all = self._sample_rect_mask(
            grid_h,
            grid_w,
            self.cfg.context_scale_min,
            self.cfg.context_scale_max,
            self.cfg.aspect_ratio_min,
            self.cfg.aspect_ratio_max,
            device,
            num_masks=batch_size,
        )  # (B, N)

        # Sample all target masks at once for each target block
        target_masks: List[torch.Tensor] = []
        for block_idx in range(self.cfg.num_target_blocks):
            target_masks_all = self._sample_rect_mask(
                grid_h,
                grid_w,
                self.cfg.target_scale_min,
                self.cfg.target_scale_max,
                self.cfg.aspect_ratio_min,
                self.cfg.aspect_ratio_max,
                device,
                num_masks=batch_size,
            )  # (B, N)

            # Apply context mask constraint: target cannot overlap context
            available = ~context_masks_all  # (B, N)
            target_masks_all = target_masks_all & available

            # Handle empty target masks (vectorized, no Python loops)
            empty_mask = target_masks_all.sum(dim=1) == 0  # (B,) bool

            if empty_mask.any():
                # For each empty mask, set one random token from available
                # Use einsum/advanced indexing to set mask values vectorized
                empty_indices = torch.where(empty_mask)[0]  # Indices of empty masks

                # For each empty batch element, sample random available position
                for idx in empty_indices:
                    avail_positions = torch.where(available[idx])[0]
                    if avail_positions.numel() > 0:
                        # Vectorized random selection (no .item() needed for indexing)
                        random_offset = torch.randint(0, avail_positions.numel(), (1,), device=device)
                        target_masks_all[idx, avail_positions[random_offset]] = True
                    else:
                        # Fallback: sample any position
                        random_pos = torch.randint(0, grid_n, (1,), device=device)
                        target_masks_all[idx, random_pos] = True

            target_masks.append(target_masks_all)

        return context_masks_all, target_masks

    def _gather_padded_target_tokens(
        self,
        full_tokens: torch.Tensor,
        target_masks: List[torch.Tensor],
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Gather per-block target tokens and pad each block to a common length.
        Fully vectorized to eliminate per-batch loops.

        Returns:
            target_tokens: list[(B, max_Nt, D)]
            target_valid_masks: list[(B, max_Nt)] bool
        """
        targets: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        device = full_tokens.device
        bsz = full_tokens.shape[0]

        for mask in target_masks:
            counts = mask.sum(dim=1)  # (B,)
            max_count = int(counts.max().item())
            if max_count == 0:
                max_count = 1

            # Vectorized index gathering for all batch elements
            idx_range = torch.arange(mask.shape[1], device=device).unsqueeze(0).expand(bsz, -1)  # (B, N)

            # Sort to move True values to front
            sorted_order = torch.argsort((~mask).int(), dim=1, descending=False)  # (B, N)
            sorted_indices = torch.gather(idx_range, dim=1, index=sorted_order)  # (B, N)
            batch_indices = sorted_indices[:, :max_count]  # (B, max_count)

            # Gather tokens using advanced indexing
            # Create batch indices: (B, max_count)
            batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, max_count)  # (B, max_count)

            # Gather tokens for all batch elements at once
            tok = full_tokens[batch_idx, batch_indices, :]  # (B, max_count, D)

            # Create validity mask: True where we have actual tokens
            valid_mask = torch.arange(max_count, device=device).unsqueeze(0) < counts.unsqueeze(1)  # (B, max_count)

            targets.append(tok)
            masks.append(valid_mask)

        return targets, masks

    def _predict_targets(
        self,
        context_tokens: torch.Tensor,
        context_padding_mask: torch.Tensor | None,
        target_masks: List[torch.Tensor],
        pos_spatial: torch.Tensor,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Predict target-token latents from context tokens and target positions.

        Critical optimization: Process ALL target blocks in ONE forward pass through predictor.
        Instead of num_target_blocks separate passes, concatenate all target queries and process
        in a single batched operation. This is valid because target blocks are independent.
        """
        bsz = context_tokens.shape[0]
        device = context_tokens.device

        context_z = self.pred_in(context_tokens)  # (B, Nc, D_pred)
        pos_proj = self.pred_in_pos(pos_spatial)  # (1, N, D_pred)

        # Process all target blocks at once: gather queries and split results afterward
        all_target_queries_list: List[torch.Tensor] = []
        all_valid_masks_list: List[torch.Tensor] = []
        block_max_counts: List[int] = []
        idx_range = torch.arange(target_masks[0].shape[1], device=device).unsqueeze(0).expand(bsz, -1)

        for target_mask in target_masks:
            counts = target_mask.sum(dim=1)  # (B,)
            max_count = int(counts.max().item())
            if max_count == 0:
                max_count = 1
            block_max_counts.append(max_count)

            # Vectorized index gathering
            valid_mask = torch.arange(max_count, device=device).unsqueeze(0) < counts.unsqueeze(1)  # (B, max_count)

            sorted_order = torch.argsort((~target_mask).int(), dim=1, descending=False)
            sorted_indices = torch.gather(idx_range, dim=1, index=sorted_order)  # (B, N)
            batch_indices = sorted_indices[:, :max_count]  # (B, max_count)

            # Gather position embeddings
            pos_proj_batch = pos_proj.expand(bsz, -1, -1)  # (B, N, D_pred)
            batch_indices_expanded = batch_indices.unsqueeze(-1).expand(-1, -1, pos_proj.shape[-1])
            target_pos = torch.gather(pos_proj_batch, dim=1, index=batch_indices_expanded)  # (B, max_count, D_pred)

            # Create target queries: mask token + position embedding
            target_queries = self.pred_mask_token.expand(bsz, max_count, -1) + target_pos  # (B, max_count, D_pred)

            all_target_queries_list.append(target_queries)
            all_valid_masks_list.append(valid_mask)

        # Concatenate all target block queries into one (B, sum(max_counts), D_pred) tensor
        all_target_queries = torch.cat(all_target_queries_list, dim=1)  # (B, sum of max_counts, D)

        # Concatenate valid masks for combined padding mask
        all_valid_masks = torch.cat(all_valid_masks_list, dim=1)  # (B, sum of max_counts)

        # Concatenate context tokens and ALL target queries
        z = torch.cat([context_z, all_target_queries], dim=1)  # (B, Nc + sum(max_counts), D)

        # Build combined padding mask (True = invalid/padding)
        if context_padding_mask is not None:
            combined_padding = torch.cat([context_padding_mask, ~all_valid_masks], dim=1)
        else:
            combined_padding = torch.cat(
                [torch.zeros((bsz, context_z.shape[1]), dtype=torch.bool, device=device), ~all_valid_masks], dim=1
            )

        # CRITICAL: Single forward pass through ALL predictor blocks for ALL target blocks
        # This replaces num_target_blocks separate forward passes
        for blk in self.pred_blocks:
            z = blk(z, padding_mask=combined_padding)

        z = self.pred_norm(z)

        # Apply pred_out to ALL predictions at once (single batched operation)
        # This is critical for GPU efficiency - calling pred_out inside the loop causes oscillation
        pred_target_all = self.pred_out(z[:, context_z.shape[1] :, :])  # (B, sum(max_counts), D_embed)

        # Split predictions back into per-block tensors
        preds: List[torch.Tensor] = []
        pred_masks: List[torch.Tensor] = []
        offset = 0

        for block_idx, max_count in enumerate(block_max_counts):
            preds.append(pred_target_all[:, offset : offset + max_count, :])
            pred_masks.append(all_valid_masks_list[block_idx])
            offset += max_count

        return preds, pred_masks

    def forward(
        self,
        imgs: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_masks: List[torch.Tensor] | None = None,
    ) -> Dict[str, object]:
        """
        Forward pass for I-JEPA architecture.

        Args:
            imgs: Input images, shape (B, 3, H, W)
            context_mask: Optional pre-computed context mask (B, N). If None, samples masks.
            target_masks: Optional pre-computed target masks list[(B, N)]. If None, samples masks.

        Returns model outputs required for downstream training code:
        - predicted_target_tokens: list[(B, Nt_i, D)]
        - target_tokens: list[(B, Nt_i, D)]
        - context_tokens: (B, Nc, D)
        - context_mask: (B, N)
        - target_masks: list[(B, N)]
        """
        if imgs.ndim != 4:
            raise ValueError("imgs must have shape (B, C, H, W)")

        assert self.cfg.image_size == imgs.shape[2]
        assert self.cfg.image_size == imgs.shape[3]

        bsz = imgs.shape[0]
        device = imgs.device

        # Use provided masks or sample them on-the-fly
        if context_mask is None or target_masks is None:
            context_mask, target_masks = self._sample_context_and_target_masks(batch_size=bsz, device=device)
        else:
            # Validate pre-computed masks
            if context_mask.shape != (bsz, self.num_patches):
                raise ValueError(f"context_mask shape {context_mask.shape} != expected {(bsz, self.num_patches)}")
            if len(target_masks) != self.cfg.num_target_blocks:
                raise ValueError(f"num target_masks {len(target_masks)} != config {self.cfg.num_target_blocks}")

        context_tokens, context_padding_mask = self.context_encoder.forward_full(
            imgs,
            patch_keep_mask=context_mask,
            add_cls_token=False,
            return_padding_mask=True,
        )

        self.target_encoder.eval()
        with torch.no_grad():
            full_target_tokens = self.target_encoder.forward_full(
                imgs,
                patch_keep_mask=None,
                add_cls_token=False,
            )

        h_patch = imgs.shape[2] // self.cfg.patch_size
        w_patch = imgs.shape[3] // self.cfg.patch_size
        pos_all = self.context_encoder.interpolate_pos_encoding(h_patch, w_patch)[:, 1:, :]

        predicted_target_tokens, predicted_target_masks = self._predict_targets(
            context_tokens, context_padding_mask, target_masks, pos_all
        )

        target_tokens, target_valid_masks = self._gather_padded_target_tokens(full_target_tokens, target_masks)

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
    cfg = IJEPAConfig()
    model = I_JEPA(cfg)
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    out = model(x)
    assert "predicted_target_tokens" in out
    assert "target_tokens" in out


if __name__ == "__main__":
    _smoke_test()
