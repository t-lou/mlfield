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
    ) -> torch.Tensor:
        """Sample one rectangular boolean mask on a patch grid."""
        total = grid_h * grid_w

        area = torch.empty(1, device=device).uniform_(scale_min, scale_max).item() * total
        aspect = torch.empty(1, device=device).uniform_(ar_min, ar_max).item()

        h = int(round((area * aspect) ** 0.5))
        w = int(round((area / aspect) ** 0.5))
        h = max(1, min(h, grid_h))
        w = max(1, min(w, grid_w))

        top = torch.randint(0, grid_h - h + 1, (1,), device=device).item()
        left = torch.randint(0, grid_w - w + 1, (1,), device=device).item()
        mask_2d = torch.zeros((grid_h, grid_w), dtype=torch.bool, device=device)
        mask_2d[top : top + h, left : left + w] = True
        return mask_2d.reshape(-1)

    def _sample_context_and_target_masks(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Create one context mask and multiple target masks.

        Returns:
            context_keep_mask: (B, N) bool
            target_masks: list[(B, N) bool]
        """
        grid_h = self.grid_size
        grid_w = self.grid_size

        context_keep = self._sample_rect_mask(
            grid_h,
            grid_w,
            self.cfg.context_scale_min,
            self.cfg.context_scale_max,
            self.cfg.aspect_ratio_min,
            self.cfg.aspect_ratio_max,
            device,
        )

        available = ~context_keep
        target_masks_1d: List[torch.Tensor] = []
        for _ in range(self.cfg.num_target_blocks):
            target = self._sample_rect_mask(
                grid_h,
                grid_w,
                self.cfg.target_scale_min,
                self.cfg.target_scale_max,
                self.cfg.aspect_ratio_min,
                self.cfg.aspect_ratio_max,
                device,
            )
            target = target & available

            if target.sum() == 0:
                avail_idx = torch.where(available)[0]
                if avail_idx.numel() == 0:
                    break
                target = torch.zeros_like(available)
                target[avail_idx[torch.randint(0, avail_idx.numel(), (1,), device=device)]] = True

            target_masks_1d.append(target)

        if len(target_masks_1d) == 0:
            # Minimal safety fallback: at least one target token
            fallback = torch.zeros_like(available)
            fallback[torch.randint(0, fallback.numel(), (1,), device=device)] = True
            target_masks_1d = [fallback]

        context_keep_mask = context_keep.unsqueeze(0).repeat(batch_size, 1)
        target_masks = [tm.unsqueeze(0).repeat(batch_size, 1) for tm in target_masks_1d]
        return context_keep_mask, target_masks

    def _predict_targets(
        self,
        context_tokens: torch.Tensor,
        target_masks: List[torch.Tensor],
        pos_spatial: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Predict target-token latents from context tokens and target positions."""
        bsz = context_tokens.shape[0]
        context_z = self.pred_in(context_tokens)

        preds: List[torch.Tensor] = []
        pos_proj = self.pred_in_pos(pos_spatial)

        for target_mask in target_masks:
            idx = torch.where(target_mask[0])[0]  # target_masks is assumed to be the same per batch
            target_queries = self.pred_mask_token.repeat(bsz, idx.numel(), 1)
            target_queries = target_queries + pos_proj[:, idx, :]  # alternative + pos_spatial[:, idx, :]

            z = torch.cat([context_z, target_queries], dim=1)
            for blk in self.pred_blocks:
                z = blk(z)
            z = self.pred_norm(z)

            pred_target = self.pred_out(z[:, -idx.numel() :, :])
            preds.append(pred_target)

        return preds

    def forward(self, imgs: torch.Tensor) -> Dict[str, object]:
        """
        Forward pass for I-JEPA architecture.

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

        context_mask, target_masks = self._sample_context_and_target_masks(batch_size=bsz, device=device)

        context_tokens = self.context_encoder.forward_full(
            imgs,
            patch_keep_mask=context_mask,
            add_cls_token=False,
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

        predicted_target_tokens = self._predict_targets(context_tokens, target_masks, pos_all)

        target_tokens: List[torch.Tensor] = []
        for mask in target_masks:
            idx = torch.where(mask[0])[0]
            target_tokens.append(full_target_tokens[:, idx, :])

        return {
            "predicted_target_tokens": predicted_target_tokens,
            "target_tokens": target_tokens,
            "context_tokens": context_tokens,
            "context_mask": context_mask,
            "target_masks": target_masks,
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
