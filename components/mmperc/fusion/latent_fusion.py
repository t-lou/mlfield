from __future__ import annotations

import torch
import torch.nn as nn

from components.vit.vit_block import VitBlock

# Fixed modality vocabulary for the fusion latent space. Extend here (not by
# adding new __init__ args) if a new modality is added later.
MODALITIES = ("lidar", "camera", "can")


class CrossAttentionBlock(nn.Module):
    """
    Perceiver-IO-style cross-attention: a small set of latent queries attends
    into a (potentially large) set of context tokens. Cost is O(M*N) for the
    attention itself (M = num latents, N = total context tokens) rather than
    the O(N^2) of full self-attention over everything concatenated together —
    this is the actual point of using latents as queries instead of just
    prepending them into one self-attended sequence.

    Pre-norm, residual, single fused MLP after attention — same block shape
    as VitBlock, just with separate query/context inputs instead of one x.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        latents: (B, M, D) — learned query set (small, fixed size)
        context: (B, N, D) — concatenated modality tokens (large, variable size)
        """
        B, M, D = latents.shape
        q = self.to_q(self.norm_q(latents))
        kv = self.to_kv(self.norm_kv(context))
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, M, D)
        out = self.proj_drop(self.proj(out))

        latents = latents + out
        latents = latents + self.mlp(self.norm_mlp(latents))
        return latents


class LatentFusion(nn.Module):
    """
    Multimodal JEPA fusion, LeJEPA-style.

    Structure:
        - a small fixed set of learned latent queries cross-attend into the
          concatenated (lidar, camera, can) token streams — Perceiver-style,
          avoids O(N^2) self-attention over the full multimodal token count.
        - a few rounds of self-attention among the latents only (cheap, since
          the latent set is small) to let them exchange information.

    Deliberately NOT included, per LeJEPA (Balestriero & LeCun, 2025):
        - no momentum/EMA teacher copy of this module
        - no stop-gradient on any branch
    Collapse is prevented by a SIGReg regularization term applied to this
    module's *output* embeddings in the training loss, not by architecture
    here. The same LatentFusion instance is meant to be called on both the
    context (masked) view and the target (full) view — see forward() note.
    Implement SIGReg itself by consulting the official repo
    (rbalestr-lab/lejepa) rather than guessing the exact statistic; it is not
    reproduced here.

    Each modality's tokens must already be projected to `dim` before calling
    forward (e.g. MultiCameraEncoder.out_channels, BEVTokenizer's embed_dim,
    and CANEncoder's dim must all equal `dim`).
    """

    def __init__(
        self,
        dim: int,
        num_latents: int = 32,
        num_heads: int = 8,
        latent_self_attn_depth: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.modality_embed = nn.Embedding(len(MODALITIES), dim)

        self.cross_attn = CrossAttentionBlock(dim, num_heads, mlp_ratio=mlp_ratio)
        self.latent_self_attn = nn.ModuleList(
            [VitBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(latent_self_attn_depth)]
        )
        self.norm_out = nn.LayerNorm(dim)

    def _add_modality_embed(self, tokens: torch.Tensor, modality: str) -> torch.Tensor:
        idx = MODALITIES.index(modality)
        return tokens + self.modality_embed.weight[idx]

    def forward(
        self,
        lidar_tokens: torch.Tensor,
        cam_tokens: torch.Tensor,
        can_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        lidar_tokens: (B, N_lidar, dim) — e.g. from a BEV tokenizer
        cam_tokens:   (B, N_cam, dim)   — concatenated MultiCameraEncoder output
        can_tokens:   (B, 1, dim) or None — CANEncoder output; ego-state only,
                      never action-adjacent signals (see prior discussion)

        Returns:
            latents: (B, num_latents, dim)

        Call this once on a masked/context view and once on the full/target
        view (same weights, no stop-gradient) to get the pair the predictor
        and SIGReg loss operate on.
        """
        B = lidar_tokens.shape[0]

        lidar_tokens = self._add_modality_embed(lidar_tokens, "lidar")
        cam_tokens = self._add_modality_embed(cam_tokens, "camera")
        context = [lidar_tokens, cam_tokens]

        if can_tokens is not None:
            can_tokens = self._add_modality_embed(can_tokens, "can")
            context.append(can_tokens)

        context = torch.cat(context, dim=1)  # (B, N_total, dim)

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        latents = self.cross_attn(latents, context)

        for blk in self.latent_self_attn:
            latents = blk(latents)

        return self.norm_out(latents)


def _smoke_test():
    B, dim = 2, 128
    fusion = LatentFusion(dim=dim, num_latents=32, num_heads=8)

    lidar_tokens = torch.randn(B, 512, dim)
    cam_tokens = torch.randn(B, 6 * 64, dim)  # e.g. 6 cameras x 64 tokens each
    can_tokens = torch.randn(B, 1, dim)

    latents = fusion(lidar_tokens, cam_tokens, can_tokens)
    assert latents.shape == (B, 32, dim), latents.shape

    # can_tokens optional
    latents_no_can = fusion(lidar_tokens, cam_tokens, can_tokens=None)
    assert latents_no_can.shape == (B, 32, dim)

    print("LatentFusion smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
