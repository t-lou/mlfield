from __future__ import annotations

import torch
from torch import Tensor, nn

from components.mmperc.fusion.perceiver_fusion import (
    CrossAttention,
    FeedForward,
    LatentSelfAttentionLayer,
    ModalityRegistry,
    ModalitySpec,
)


class LatentFusion(nn.Module):
    """
    Multimodal JEPA fusion, built on the same Perceiver primitives as
    PerceiverFusionBlock (perceiver_core.py) rather than a separate,
    weaker reimplementation. Concretely this backports, vs. the first draft:
      - F.scaled_dot_product_attention instead of manual matmul+softmax
      - learned per-attention logit_scale (temperature) -- matters here at
        least as much as in PerceiverFusionBlock: lidar tokens (a full BEV
        grid) and camera tokens (a per-camera dict) will differ in count by
        orders of magnitude, and an uncalibrated softmax can let the larger
        modality dominate attention mass and starve the smaller one
      - true interleaved Perceiver structure: (cross-attn into all input
        tokens, then latent self-attn), repeated `depth` times, so latents
        can re-query the input after refining themselves -- not a single
        cross-attn pass followed by self-attn-only refinement
      - share_weights, the standard Perceiver parameter-saving trick
      - a real modality registry (ModalityRegistry/ModalitySpec/add_modality)
        instead of a hardcoded 3-tuple of modality names

    Per-camera modality registration (see camera_names below) also fixes the
    "single cam_hw tuple" limitation that both MultiCameraEncoder's caller
    and PerceiverFusionBlock's default 2-modality setup still carry: each
    camera gets its own registered modality and its own spatial hint,
    instead of one hint describing a concatenated multi-camera sequence.

    No lidar tokenizer is needed separately: ModalityEncoder's "grid" kind
    already does project + position-embed + flatten for a (B, C, H, W) input,
    so PointPillarBEV's raw BEV grid can be passed here directly.

    LeJEPA-style (Balestriero & LeCun, 2025): NO momentum/EMA teacher copy of
    this module, NO stop-gradient on any branch. The same instance is meant
    to be called on both the context (masked) view and the target (full)
    view. Collapse is prevented by a SIGReg regularization term applied to
    this module's *output* embeddings in the training loss, not by anything
    architectural here -- implement SIGReg by consulting the official repo
    (rbalestr-lab/lejepa) rather than guessing the exact statistic; it is
    not reproduced in this module.

    Open question, not resolved here: MultiCameraEncoder already adds a
    per-camera cam_id_embed before fusion, and this module's ModalityEncoder
    adds its own per-camera modality_embed on top. That's two learned
    camera-identity signals at two stages -- possibly fine (harmless
    redundancy, or usefully layered), possibly worth dropping one. Flagging
    rather than deciding silently.
    """

    def __init__(
        self,
        dim: int,
        camera_names: list[str],
        camera_channels: int,
        lidar_channels: int,
        can_channels: int | None = None,
        num_latents: int = 32,
        num_heads: int = 8,
        depth: int = 2,
        share_weights: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.share_weights = share_weights
        self.camera_names = list(camera_names)

        modalities = [ModalitySpec(name="lidar", in_channels=lidar_channels, kind="grid")]
        modalities += [
            ModalitySpec(name=f"camera_{cam_id}", in_channels=camera_channels, kind="seq")
            for cam_id in self.camera_names
        ]
        if can_channels is not None:
            modalities.append(ModalitySpec(name="can", in_channels=can_channels, kind="seq"))

        self._registry = ModalityRegistry(modalities, dim)

        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * dim**-0.5)

        n_modules = 1 if share_weights else depth
        self.encode_norm_latent = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_modules)])
        self.encode_attn = nn.ModuleList(
            [CrossAttention(dim, num_heads, dropout, logit_scale_init=4.0) for _ in range(n_modules)]
        )
        self.encode_norm_ffn = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_modules)])
        self.encode_ffn = nn.ModuleList([FeedForward(dim) for _ in range(n_modules)])
        self.self_layers = nn.ModuleList([LatentSelfAttentionLayer(dim, num_heads, dropout) for _ in range(n_modules)])

        self.norm_out = nn.LayerNorm(dim)

    @property
    def modality_encoders(self) -> nn.ModuleDict:
        return self._registry.modality_encoders

    def add_modality(self, spec: ModalitySpec) -> None:
        """Register an additional input modality after construction (e.g. radar)."""
        self._registry.add_modality(spec)

    def forward(
        self,
        lidar_bev: Tensor,
        camera_tokens: dict[str, Tensor],
        can_tokens: Tensor | None = None,
        cam_hw: dict[str, tuple[int, int]] | None = None,
    ) -> Tensor:
        """
        lidar_bev:     (B, C, H, W) -- raw BEV feature grid, e.g. from PointPillarBEV
        camera_tokens: {camera_id: (B, N_cam_i, C)} -- e.g. MultiCameraEncoder's
                       tokens_per_cam output. Every key must have been registered
                       via camera_names at construction (or add_modality after).
        can_tokens:    (B, 1, C) or None -- CANEncoder output; ego-state only,
                       never action-adjacent signals.
        cam_hw:        {camera_id: (H', W')} -- per-camera spatial hint, e.g. from
                       each camera's feat map shape. Optional; omit for a camera
                       to fall back to 1D positional encoding for its tokens.

        Returns:
            latents: (B, num_latents, dim)

        Call this once on a masked/context view and once on the full/target
        view (same weights, no stop-gradient) to get the pair the predictor
        and SIGReg loss operate on.
        """
        inputs: dict[str, Tensor] = {"lidar": lidar_bev}
        spatial_hints: dict[str, tuple[int, int]] = {}

        for cam_id, tokens in camera_tokens.items():
            name = f"camera_{cam_id}"
            if name not in self.modality_encoders:
                raise KeyError(
                    f"camera '{cam_id}' was not registered at construction "
                    f"(camera_names={self.camera_names}); add it via add_modality first."
                )
            inputs[name] = tokens
            if cam_hw is not None and cam_id in cam_hw:
                spatial_hints[name] = cam_hw[cam_id]

        if can_tokens is not None:
            inputs["can"] = can_tokens

        tokens = self._registry.encode(inputs, spatial_hints)  # (B, N_total, dim)

        B = tokens.shape[0]
        latents = self.latents.expand(B, -1, -1)
        for i in range(self.depth):
            j = 0 if self.share_weights else i

            q = self.encode_norm_latent[j](latents)
            latents = latents + self.encode_attn[j](q, tokens, training=self.training)
            latents = latents + self.encode_ffn[j](self.encode_norm_ffn[j](latents))

            latents = self.self_layers[j](latents, training=self.training)

        return self.norm_out(latents)


def _smoke_test():
    B, dim = 2, 128
    camera_names = ["front_left", "front_center", "front_right"]

    fusion = LatentFusion(
        dim=dim,
        camera_names=camera_names,
        camera_channels=dim,
        lidar_channels=dim,
        can_channels=dim,
        num_latents=32,
        num_heads=8,
        depth=2,
        share_weights=True,
    )

    lidar_bev = torch.randn(B, dim, 32, 32)
    camera_tokens = {cam_id: torch.randn(B, 64, dim) for cam_id in camera_names}
    cam_hw = {cam_id: (8, 8) for cam_id in camera_names}
    can_tokens = torch.randn(B, 1, dim)

    latents = fusion(lidar_bev, camera_tokens, can_tokens, cam_hw=cam_hw)
    assert latents.shape == (B, 32, dim), latents.shape

    # can_tokens optional
    latents_no_can = fusion(lidar_bev, camera_tokens, can_tokens=None, cam_hw=cam_hw)
    assert latents_no_can.shape == (B, 32, dim)

    # Unregistered camera should raise, not silently drop.
    try:
        fusion(lidar_bev, {"unregistered_cam": torch.randn(B, 64, dim)})
        raise AssertionError("expected KeyError for unregistered camera")
    except KeyError:
        pass

    n_params = sum(p.numel() for p in fusion.parameters())
    print(f"LatentFusion smoke test passed. Params: {n_params:,}")


if __name__ == "__main__":
    _smoke_test()
