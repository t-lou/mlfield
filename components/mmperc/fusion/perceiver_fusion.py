from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams
from components.vit.position_embedding import PosEmbdCache

# --------------------------------------------------------------------------- #
# Generic attention / FFN building blocks
# --------------------------------------------------------------------------- #


class CrossAttention(nn.Module):
    """
    q attends into (k, v). Always dispatches through SDPA (flash / mem-efficient
    when available), so this never materializes an O(N_q * N_kv) weight matrix
    in eager Python — same memory profile as the original FuTr block.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, kv_dim: int | None = None):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        kv_dim = kv_dim if kv_dim is not None else dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(kv_dim, dim)
        self.v_proj = nn.Linear(kv_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q_in: Tensor, kv_in: Tensor, training: bool = False) -> Tensor:
        B, Nq, _ = q_in.shape
        Nkv = kv_in.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(q_in).view(B, Nq, H, Dh).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, Nkv, H, Dh).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, Nkv, H, Dh).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if training else 0.0)
        out = out.transpose(1, 2).reshape(B, Nq, H * Dh)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Simple 2-layer MLP with GELU activation, used in the Perceiver latent self-attention stack.
    """

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LatentSelfAttentionLayer(nn.Module):
    """
    One pre-norm transformer layer used inside the latent self-attention stack.
    Cost is O(M^2 * C) with M = num_latents, which is negligible next to the
    O(M * N_input * C) encode cross-attention as long as M stays small (16-64).
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, dropout)  # q=kv=latents
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, latents: Tensor, training: bool = False) -> Tensor:
        x = self.norm1(latents)
        latents = latents + self.attn(x, x, training=training)
        latents = latents + self.ffn(self.norm2(latents))
        return latents


# --------------------------------------------------------------------------- #
# Modality registry: this is the extensibility hook. Adding a new sensor
# (radar, lidar, future cameras, ...) is a one-line `add_modality` call plus
# passing the tensor into `forward(inputs={...})` — no other code changes.
# --------------------------------------------------------------------------- #


@dataclass
class ModalitySpec:
    name: str
    in_channels: int
    kind: Literal["grid", "seq"]  # "grid": (B, C, H, W)  |  "seq": (B, N, C)
    # for "seq" modalities whose tokens correspond to an (h, w) grid (e.g. a camera
    # feature map flattened to tokens), pass a fixed shape here, or leave None and
    # supply cam_hw-style spatial hints per-call via forward(spatial_hints=...).
    grid_shape: tuple[int, int] | None = None


class ModalityEncoder(nn.Module):
    """Projects a single modality's raw tokens into the shared latent width,
    adds positional encoding + a learned modality-type embedding (so the
    latents can tell tokens from different sensors apart, Perceiver-IO style),
    and flattens grid modalities into a token sequence."""

    def __init__(self, spec: ModalitySpec, latent_dim: int, pos_cache: PosEmbdCache, modality_index: int):
        super().__init__()
        self.spec = spec
        self.proj = nn.Linear(spec.in_channels, latent_dim)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        # Learned per-modality embedding, added on top of positional encoding.
        self.modality_embed = nn.Parameter(torch.randn(1, 1, latent_dim) * latent_dim**-0.5)
        self._pos_cache = pos_cache
        self._modality_index = modality_index

    def forward(self, x: Tensor, spatial_hint: tuple[int, int] | None = None) -> Tensor:
        spec = self.spec
        if spec.kind == "grid":
            B, C, H, W = x.shape
            tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            tokens = self.proj(tokens)
            pos = self._pos_cache.get_2d(H, W, tokens.shape[-1], x.device, x.dtype)
        else:
            B, N, C = x.shape
            tokens = self.proj(x)
            hw = spatial_hint if spatial_hint is not None else spec.grid_shape
            if hw is not None:
                assert hw[0] * hw[1] == N, f"{spec.name}: grid_shape {hw} does not match N={N}"
                pos = self._pos_cache.get_2d(hw[0], hw[1], tokens.shape[-1], x.device, x.dtype)
            else:
                pos = self._pos_cache.get_1d(N, tokens.shape[-1], x.device, x.dtype)

        tokens = tokens + self.pos_scale * pos.unsqueeze(0) + self.modality_embed
        return tokens


# --------------------------------------------------------------------------- #
# Output heads: the other extensibility hook. Today we only need a single
# global FiLM (scale, shift) vector to modulate BEV, but future outputs
# (per-camera predictions, per-pixel decode, auxiliary heads, ...) can be
# added as additional named OutputQuery heads reading from the same latents,
# following the Perceiver-IO decoder pattern.
# --------------------------------------------------------------------------- #


class OutputQueryHead(nn.Module):
    """
    A learnable query (or set of queries) that cross-attends into the final
    latent array to decode one named output. `num_queries=1` + a linear head
    reproduces the current FiLM read-out; `num_queries=HW` with a per-pixel
    query positional encoding would let this decode straight back to BEV
    resolution later, without touching the encoder side at all.
    """

    def __init__(self, dim: int, out_dim: int, num_heads: int, num_queries: int = 1, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, dim) * dim**-0.5)
        self.attn = CrossAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, latents: Tensor, training: bool = False) -> Tensor:
        B = latents.shape[0]
        q = self.query.expand(B, -1, -1)
        out = self.attn(q, latents, training=training)
        out = self.norm(out)
        return self.head(out)


# --------------------------------------------------------------------------- #
# Main block
# --------------------------------------------------------------------------- #


class PerceiverFusionBlock(nn.Module):
    """
    Perceiver-style multi-modal fusion, drop-in-compatible with FuTrFusionBlock
    for the (bev, camera) -> fused_bev use case, but built around three
    extensibility seams:

      1. Modalities are registered (`modalities=[ModalitySpec(...), ...]`) and
         fed as a dict at forward time, so adding radar/lidar/extra cameras
         later needs no architectural change.
      2. The core is a genuine Perceiver stack: latents cross-attend into the
         concatenated input tokens, then run through a small self-attention
         transformer, repeated `depth` times with optional weight sharing
         (`share_weights=True`, the standard Perceiver parameter-saving trick).
      3. Outputs are named `OutputQueryHead`s reading from the final latents,
         so new output types (per-camera heads, auxiliary losses, a future
         per-pixel decode back to BEV resolution) are additive.

    Complexity, with M = num_latents, N = total input tokens (HW + camera + ...):
      encode cross-attn : O(M * N * C)   -- dominant term, same order as FuTr's
                                            O(N_cam * HW * C) when M ~ N_cam
      latent self-attn  : O(depth * M^2 * C) -- negligible for M in [16, 64]
      readout           : O(M * C)        -- cheaper than FuTr's O(N_cam * C) pool
    """

    def __init__(
        self,
        params: MmpercParams,
        modalities: list[ModalitySpec] | None = None,
        num_latents: int = 32,
        num_heads: int = 4,
        depth: int = 2,
        share_weights: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        C = params.bev_params.bev_channels
        self.latent_dim = C
        self.depth = depth
        self.dropout_p = dropout

        if modalities is None:
            # Default registration matching the original FuTrFusionBlock inputs.
            modalities = [
                ModalitySpec(name="bev", in_channels=C, kind="grid"),
                ModalitySpec(name="camera", in_channels=C, kind="seq"),
            ]

        self._pos_cache = PosEmbdCache()
        self.modality_encoders = nn.ModuleDict()
        for i, spec in enumerate(modalities):
            self.modality_encoders[spec.name] = ModalityEncoder(spec, C, self._pos_cache, i)

        # Learnable latent array — the Perceiver bottleneck.
        self.latents = nn.Parameter(torch.randn(1, num_latents, C) * C**-0.5)

        # Encode: latents <- input tokens. One cross-attn module per depth step,
        # unless share_weights, in which case a single module is reused —
        # the standard Perceiver trick for keeping parameter count roughly
        # constant in depth.
        n_encode_modules = 1 if share_weights else depth
        self.encode_norm_latent = nn.ModuleList([nn.LayerNorm(C) for _ in range(n_encode_modules)])
        self.encode_attn = nn.ModuleList([CrossAttention(C, num_heads, dropout) for _ in range(n_encode_modules)])
        self.encode_norm_ffn = nn.ModuleList([nn.LayerNorm(C) for _ in range(n_encode_modules)])
        self.encode_ffn = nn.ModuleList([FeedForward(C) for _ in range(n_encode_modules)])

        n_self_modules = 1 if share_weights else depth
        self.self_layers = nn.ModuleList(
            [LatentSelfAttentionLayer(C, num_heads, dropout) for _ in range(n_self_modules)]
        )
        self.share_weights = share_weights

        # Output heads. Registered in a dict so new ones can be added later
        # (e.g. `block.output_heads["aux_seg"] = OutputQueryHead(...)`).
        self.output_heads = nn.ModuleDict(
            {
                "bev_film": OutputQueryHead(C, out_dim=C * 2, num_heads=num_heads, num_queries=1, dropout=dropout),
            }
        )

    def add_modality(self, spec: ModalitySpec) -> None:
        """Register an additional input modality after construction."""
        idx = len(self.modality_encoders)
        self.modality_encoders[spec.name] = ModalityEncoder(spec, self.latent_dim, self._pos_cache, idx)

    def add_output_head(self, name: str, head: OutputQueryHead) -> None:
        """Register an additional named output, decoded from the same shared latents."""
        self.output_heads[name] = head

    def encode(self, inputs: dict[str, Tensor], spatial_hints: dict[str, tuple[int, int]] | None = None) -> Tensor:
        """
        Runs every registered modality present in `inputs` through its encoder
        and concatenates the resulting token sequences. Modalities absent from
        `inputs` (e.g. a sensor that's temporarily unavailable) are simply skipped.
        """
        spatial_hints = spatial_hints or {}
        token_chunks = []
        for name, encoder in self.modality_encoders.items():
            if name not in inputs:
                continue
            token_chunks.append(encoder(inputs[name], spatial_hint=spatial_hints.get(name)))
        assert token_chunks, "PerceiverFusionBlock.encode: no registered modality found in `inputs`"
        return torch.cat(token_chunks, dim=1)  # (B, N_total, C)

    def forward(
        self,
        bev: Tensor,
        camera: Tensor,
        cam_hw: tuple[int, int] | None = None,
        extra_inputs: dict[str, Tensor] | None = None,
        extra_spatial_hints: dict[str, tuple[int, int]] | None = None,
    ) -> Tensor:
        """
        bev:    (B, C, H, W)
        camera: (B, N_cam, C)
        cam_hw: (H_cam, W_cam) or None -- spatial hint for the camera tokens.
        extra_inputs / extra_spatial_hints: optional additional modalities
            registered via `add_modality`, keyed by name. This is the seam
            future sensors plug into without changing this signature's
            required arguments.

        Returns fused_bev: (B, C, H, W), same shape as `bev`.
        """
        B, C, H, W = bev.shape

        inputs = {"bev": bev, "camera": camera}
        if extra_inputs:
            inputs.update(extra_inputs)
        spatial_hints = {"camera": cam_hw} if cam_hw is not None else {}
        if extra_spatial_hints:
            spatial_hints.update(extra_spatial_hints)

        tokens = self.encode(inputs, spatial_hints)  # (B, N_total, C)

        latents = self.latents.expand(B, -1, -1)
        for i in range(self.depth):
            j = 0 if self.share_weights else i

            # Cross-attn: latents <- input tokens (pre-norm on the latent side only,
            # matching Perceiver-IO's asymmetric encode attention).
            q = self.encode_norm_latent[j](latents)
            latents = latents + self.encode_attn[j](q, tokens, training=self.training)
            latents = latents + self.encode_ffn[j](self.encode_norm_ffn[j](latents))

            # Latent self-attention stack.
            latents = self.self_layers[j](latents, training=self.training)

        # Read out FiLM parameters for BEV modulation.
        film = self.output_heads["bev_film"](latents, training=self.training)  # (B, 1, 2C)
        scale, shift = film.squeeze(1).chunk(2, dim=-1)
        scale = scale.view(B, C, 1, 1)
        shift = shift.view(B, C, 1, 1)

        fused_bev = bev * (1 + scale) + shift
        return fused_bev


def _smoke_test():
    params = MmpercParams()
    C = params.bev_params.bev_channels
    model = PerceiverFusionBlock(params=params, num_latents=32, num_heads=4, depth=2, share_weights=True)

    B, H, W = 2, 128, 128
    N_cam = 16
    bev_input = torch.randn(B, C, H, W)
    cam_input = torch.randn(B, N_cam, C)

    output = model(bev_input, cam_input)
    assert output.shape == (B, C, H, W), f"Expected shape (B, C, H, W), got {output.shape}"

    # Extensibility check: register a radar modality after construction and
    # pass it through `extra_inputs` without touching the class.
    model.add_modality(ModalitySpec(name="radar", in_channels=C, kind="seq"))
    radar_input = torch.randn(B, 8, C)
    output2 = model(bev_input, cam_input, extra_inputs={"radar": radar_input})
    assert output2.shape == (B, C, H, W)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PerceiverFusionBlock smoke test passed. Params: {n_params:,}")


if __name__ == "__main__":
    _smoke_test()
