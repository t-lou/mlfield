import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from components.vit.vit_encoder import VitEncoder
from components.vit.dino_defs import DINOConfig


class ViTBackbone(nn.Module):
    """Vision Transformer backbone for DINO, returning the CLS token embedding."""

    def __init__(self, config: DINOConfig):
        super().__init__()
        # Create a ViT Encoder (variant, depth, drop path etc. come from the config)
        self.vit = VitEncoder(config=config.vit_config())
        # Assume vit.forward(x) returns CLS only, and vit.forward_full(x) returns all tokens

    def forward(self, x):
        """Forward pass through the ViT backbone, returning the CLS token embedding."""
        feats = self.vit.forward(x)  # [B, D]
        return feats  # CLS token embedding

    def forward_full(self, x):
        """Forward pass through the ViT backbone, returning all token embeddings."""
        feats = self.vit.forward_full(x)  # [B, num_tokens, D]
        return feats  # All token embeddings


class DINOHead(nn.Module):
    """DINO projection head: MLP -> bottleneck -> L2 norm -> weight-normalized prototypes.

    The features are normalized *before* the last layer, so the output logits are cosine
    similarities to the prototypes (bounded by [-1, 1]), as in the DINO paper. Normalizing the
    logits afterwards would shrink them to ~1/sqrt(out_dim) and make every softmax near-uniform.
    """

    def __init__(self, in_dim, config: DINOConfig, nlayers=None):
        super().__init__()
        nlayers = max(config.head_nlayers if nlayers is None else nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, config.bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, config.hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU()]
            layers.append(nn.Linear(config.hidden_dim, config.bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.mlp.apply(self._init_weights)

        self.last_layer = weight_norm(nn.Linear(config.bottleneck_dim, config.out_dim, bias=False))
        weight_g = self.last_layer.parametrizations.weight.original0
        weight_g.data.fill_(1.0)
        if config.norm_last_layer:
            weight_g.requires_grad = False

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through the DINO head, returning cosine-similarity logits (float32)."""
        x = self.mlp(x)  # bottleneck features
        x = F.normalize(x.float(), dim=-1, eps=1e-6)  # keep normalization stable under AMP
        x = self.last_layer(x)  # logits
        return x.float()


class DINOModel(nn.Module):
    """DINO model combining the ViT backbone and the DINO head, returning the projected features."""

    def __init__(self, config: DINOConfig):
        super().__init__()
        self.backbone = ViTBackbone(config)
        self.head = DINOHead(self.backbone.vit.embed_dim, config=config)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

    def forward_crops(self, crops: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run a list of crops, batching consecutive crops of equal size into one forward pass.

        Equivalent to ``[self(c) for c in crops]`` but with far fewer, larger kernels
        (the multi-crop wrapper of the DINO reference implementation).
        """
        outputs: list[torch.Tensor] = []
        start = 0
        for end in range(1, len(crops) + 1):
            if end == len(crops) or crops[end].shape[-2:] != crops[start].shape[-2:]:
                group = crops[start:end]
                out = self(torch.cat(group) if len(group) > 1 else group[0])
                outputs.extend(out.split([c.shape[0] for c in group]))
                start = end
        return outputs
