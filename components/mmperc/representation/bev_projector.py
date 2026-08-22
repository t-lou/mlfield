from __future__ import annotations

from torch import Tensor, nn

from components.vit.position_embedding import PosEmbdCache


class LatentBEVDecoder(nn.Module):
    """Decode Perceiver latents into a spatial BEV feature map.

    The fusion latents are a compact, unordered set of tokens. A learned query
    for every BEV cell restores spatial structure by cross-attending to those
    tokens, while the 2D positional encoding gives each query a stable vehicle
    frame location.
    """

    def __init__(
        self,
        bev_h: int,
        bev_w: int,
        dim: int,
        out_channels: int | None = None,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if bev_h < 1 or bev_w < 1:
            raise ValueError("bev_h and bev_w must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.dim = dim
        out_channels = dim if out_channels is None else out_channels

        self.queries = nn.Parameter(torch.randn(1, bev_h * bev_w, dim) * dim**-0.5)
        self.mode_embed = nn.Embedding(2, dim)
        self.position_cache = PosEmbdCache()

        self.query_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.latent_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )

        self.output = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, kernel_size=1),
        )

    def forward(self, latent: Tensor, mode: str = "front") -> Tensor:
        """Return BEV features with shape ``(B, out_channels, bev_h, bev_w)``."""
        if latent.ndim != 3:
            raise ValueError(f"latent must have shape (B, N, C), got {tuple(latent.shape)}")
        if latent.shape[-1] != self.dim:
            raise ValueError(f"latent channel dimension must be {self.dim}, got {latent.shape[-1]}")
        if mode not in ("front", "all"):
            raise ValueError(f"Unsupported decode mode: {mode!r}")

        batch_size = latent.shape[0]
        position = self.position_cache.get_2d(
            self.bev_h,
            self.bev_w,
            self.dim,
            device=latent.device,
            dtype=latent.dtype,
        )
        mode_id = 0 if mode == "front" else 1
        queries = self.queries.expand(batch_size, -1, -1)
        queries = queries + position.to(dtype=queries.dtype).unsqueeze(0)
        queries = queries + self.mode_embed.weight[mode_id].view(1, 1, -1)

        memory = latent
        for query_norm, latent_norm, cross_attention, ffn in zip(
            self.query_norms,
            self.latent_norms,
            self.cross_attentions,
            self.ffns,
        ):
            attended, _ = cross_attention(query_norm(queries), latent_norm(memory), latent_norm(memory))
            queries = queries + attended
            queries = queries + ffn(queries)

        features = queries.transpose(1, 2).reshape(batch_size, self.dim, self.bev_h, self.bev_w)
        return self.output(features)


BEVProjector = LatentBEVDecoder
