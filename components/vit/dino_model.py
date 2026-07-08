import torch.nn as nn

from components.vit import VitEncoder
from components.vit.dino_defs import TEACHER_BASE_RES


class ViTBackbone(nn.Module):
    """Vision Transformer backbone for DINO, returning the CLS token embedding."""

    def __init__(self, base_res=TEACHER_BASE_RES):
        super().__init__()
        # Create a ViT Encoder
        self.vit = VitEncoder(
            base_res=base_res,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            qkv_bias=True,
        )
        # assume vit.forward(x) returns CLS + patch tokens

    def forward(self, x):
        feats = self.vit.forward(x)  # [B, D]
        return feats  # CLS token embedding


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, nlayers=3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(nlayers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
            dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        """Forward pass through the DINO head, returning normalized logits."""
        x = self.mlp(x)  # features
        x = self.last_layer(x)  # logits
        x = nn.functional.normalize(x.float(), dim=-1, eps=1e-6)  # keep normalization stable under AMP
        return x


class DINOModel(nn.Module):
    """DINO model combining the ViT backbone and the DINO head, returning the projected features."""

    def __init__(self, base_res, out_dim=65536):
        super().__init__()
        self.backbone = ViTBackbone(base_res=base_res)
        dim = dim = self.backbone.vit.embed_dim
        self.head = DINOHead(dim, out_dim=out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits
