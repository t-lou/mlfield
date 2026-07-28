import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from components.definitions.mmperc_params import MmpercParams


class FuTrFusionBlock(nn.Module):
    """
    FuTrFusionBlock implements a fusion block that combines BEV (Bird's Eye View) features with camera features using
    cross-attention and FiLM-style modulation.
    """

    def __init__(self, params: MmpercParams, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        C = params.bev_params.bev_channels

        assert C % num_heads == 0, "C must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = C // num_heads
        self.dropout_p = dropout

        # Project BEV tokens (C) → (C)
        self.bev_proj = nn.Linear(C, C)

        # Project camera tokens (C) → (C)
        self.cam_proj = nn.Linear(C, C)

        # Separate K/V proj for BEV (keys/values) and Q proj for camera (queries),
        # replacing nn.MultiheadAttention's internal in-proj + out-proj.
        self.q_proj = nn.Linear(C, C)
        self.k_proj = nn.Linear(C, C)
        self.v_proj = nn.Linear(C, C)
        self.out_proj = nn.Linear(C, C)

        # FFN on camera tokens
        self.ffn = nn.Sequential(
            nn.Linear(C, C * 4),
            nn.ReLU(inplace=True),
            nn.Linear(C * 4, C),
        )

        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)

        # Fused scale+shift projection: one matmul instead of two
        self.to_film = nn.Linear(C, C * 2)

        # Learnable scales keep positional bias flexible while adding
        # effectively no memory overhead.
        self.bev_pos_scale = nn.Parameter(torch.tensor(1.0))
        self.cam_pos_scale = nn.Parameter(torch.tensor(1.0))

        # Learnable pooling query, replaces mean(dim=1)
        self.pool_query = nn.Parameter(torch.randn(1, 1, C) * (C**-0.5))
        self.pool_proj_k = nn.Linear(C, C)
        self.pool_proj_v = nn.Linear(C, C)

        self._flat_pos_cache = {}
        self._bev_pos_cache = {}

    @staticmethod
    def _positional_encoding_1d(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Generate 1D positional encoding for a sequence of length `length` with embedding dimension `dim`.
        """
        half = dim // 2
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        omega = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / max(half, 1))
        angles = pos * omega.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[1]))
        return emb[:, :dim].to(dtype=dtype)

    @staticmethod
    def _positional_encoding_2d(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Generate 2D positional encoding for a grid of size (h, w) with embedding dimension dim.
        """
        half = dim // 2
        dim_h = half
        dim_w = dim - half

        emb_h = FuTrFusionBlock._positional_encoding_1d(h, dim_h, device=device, dtype=dtype)[:, None, :]
        emb_w = FuTrFusionBlock._positional_encoding_1d(w, dim_w, device=device, dtype=dtype)[None, :, :]

        emb_h = emb_h.expand(h, w, dim_h)
        emb_w = emb_w.expand(h, w, dim_w)
        emb = torch.cat([emb_h, emb_w], dim=-1)
        return emb.view(h * w, dim)

    def _get_bev_pos(self, H, W, C, device, dtype):
        """
        Get or generate 2D positional encoding for BEV features.
        """
        key = (H, W, device, dtype)
        if key not in self._bev_pos_cache:
            self._bev_pos_cache[key] = self._positional_encoding_2d(H, W, C, device, dtype)
        return self._bev_pos_cache[key]

    def _get_flat_pos(self, length, C, device, dtype):
        """
        Get or generate 1D positional encoding for a sequence of length `length` with embedding dimension `dim`.
        """
        key = (length, device, dtype)
        if key not in self._flat_pos_cache:
            self._flat_pos_cache[key] = self._positional_encoding_1d(length, C, device, dtype)
        return self._flat_pos_cache[key]

    def _cross_attn(self, cam_tokens: Tensor, bev_tokens: Tensor) -> Tensor:
        """
        Cross-attention from camera tokens to BEV tokens.
        """
        B, N_cam, C = cam_tokens.shape
        HW = bev_tokens.shape[1]
        H = self.num_heads
        Dh = self.head_dim

        q = self.q_proj(cam_tokens).view(B, N_cam, H, Dh).transpose(1, 2)  # (B,H,N_cam,Dh)
        k = self.k_proj(bev_tokens).view(B, HW, H, Dh).transpose(1, 2)  # (B,H,HW,Dh)
        v = self.v_proj(bev_tokens).view(B, HW, H, Dh).transpose(1, 2)  # (B,H,HW,Dh)

        # Dispatches to flash / mem-efficient kernel automatically when available.
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )  # (B,H,N_cam,Dh)

        attn_out = attn_out.transpose(1, 2).reshape(B, N_cam, C)
        return self.out_proj(attn_out)

    def _attn_pool(self, cam_fused: Tensor) -> Tensor:
        """
        Pool camera tokens using attention. It takes the fused camera tokens and computes a weighted sum using
        attention, resulting in a single global camera feature.
        """
        B, N_cam, C = cam_fused.shape
        q = self.pool_query.expand(B, -1, -1)  # (B, 1, C)
        k = self.pool_proj_k(cam_fused)  # (B, N_cam, C)
        v = self.pool_proj_v(cam_fused)  # (B, N_cam, C)
        pooled = F.scaled_dot_product_attention(q, k, v)  # (B, 1, C)
        return pooled.squeeze(1)

    def forward(self, bev: Tensor, camera: Tensor, cam_hw: tuple[int, int] | None = None) -> Tensor:
        """
        bev:    (B, C, H, W)
        camera: (B, N_cam, C)
        cam_hw: (H_cam, W_cam) or None, the spatial dimensions of the camera feature map.
            If None, 1D positional encoding is used.
        """
        B, C, H, W = bev.shape

        # Flatten BEV → tokens
        bev_tokens = bev.flatten(2).transpose(1, 2)  # (B, HW, C)
        bev_tokens = self.bev_proj(bev_tokens)
        bev_pos = self._get_bev_pos(H, W, C, device=bev.device, dtype=bev.dtype)
        bev_tokens = bev_tokens + self.bev_pos_scale * bev_pos.unsqueeze(0)

        # Project camera tokens
        cam_tokens = self.cam_proj(camera)  # (B, N_cam, C)
        if cam_hw is not None:
            assert cam_hw[0] * cam_hw[1] == cam_tokens.shape[1]
            cam_pos = self._get_bev_pos(cam_hw[0], cam_hw[1], C, device=bev.device, dtype=bev.dtype)
        else:
            cam_pos = self._get_flat_pos(cam_tokens.shape[1], C, device=bev.device, dtype=bev.dtype)
        cam_tokens = cam_tokens + self.cam_pos_scale * cam_pos.unsqueeze(0)

        # need_weights=False lets PyTorch dispatch to the fused
        # scaled_dot_product_attention kernel (flash / mem-efficient attn)
        # instead of materializing the full (B, heads, N_cam, HW) weight
        # matrix — this is the main memory/runtime cost of this block.
        attn_out = self._cross_attn(cam_tokens, bev_tokens)

        # Residual + norm
        cam_fused = self.norm1(cam_tokens + attn_out)

        # FFN + residual + norm
        cam_fused = self.norm2(cam_fused + self.ffn(cam_fused))

        # Aggregate camera tokens → a single global camera feature with attention pooling
        cam_global = self._attn_pool(cam_fused)  # (B, C)

        # Convert to scale/shift for BEV modulation
        scale, shift = self.to_film(cam_global).chunk(2, dim=-1)
        scale = scale.view(B, C, 1, 1)
        shift = shift.view(B, C, 1, 1)

        # FiLM-style modulation
        fused_bev = bev * (1 + scale) + shift

        return fused_bev


def _smoke_test():
    """
    Smoke test for FuTrFusionBlock to ensure it runs without errors.
    """
    params = MmpercParams()
    model = FuTrFusionBlock(params=params, num_heads=4, dropout=0.1)

    # Create dummy BEV and camera inputs
    B, C, H, W = 2, params.bev_params.bev_channels, 128, 128
    N_cam = 16  # Number of camera tokens
    bev_input = torch.randn(B, C, H, W)
    cam_input = torch.randn(B, N_cam, C)

    # Forward pass
    output = model(bev_input, cam_input)
    assert output.shape == (B, C, H, W), f"Expected shape (B, C, H, W), got {output.shape}"
    print("FuTrFusionBlock smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
