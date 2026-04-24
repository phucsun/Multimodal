import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import StochasticDepth
from typing import Optional, Literal

BackboneName = Literal["resnet18", "efficientnet_b0", "convnext_tiny"]


# ── Drop-path Transformer layer ────────────────────────────────────────────────

class DropPathTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Pre-LN TransformerEncoderLayer with stochastic depth on residuals."""

    def __init__(self, drop_path_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = StochasticDepth(drop_path_rate, mode="row") if drop_path_rate > 0 else nn.Identity()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal: bool = False):
        # norm_first=True path — drop path wraps each residual branch
        x = src
        x = x + self.drop_path(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal))
        x = x + self.drop_path(self._ff_block(self.norm2(x)))
        return x


# ── Backbone builder ───────────────────────────────────────────────────────────

def _build_backbone(name: BackboneName, pretrained: bool):
    """Returns (feature_extractor, c_out) — all produce (B, c_out, 7, 7) for 224×224 input."""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(*list(m.children())[:-2]), 512

    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        return m.features, 1280                          # (B, 1280, 7, 7)

    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        return m.features, 768                           # (B, 768, 7, 7)

    raise ValueError(f"Unknown backbone {name!r}. Choose: resnet18 | efficientnet_b0 | convnext_tiny")


def _partial_unfreeze(backbone: nn.Module, name: BackboneName) -> None:
    """Unfreeze last 1-2 blocks for fine-tuning at low LR."""
    children = list(backbone.children())
    n_unfreeze = 1 if name == "resnet18" else 2
    for child in children[-n_unfreeze:]:
        for p in child.parameters():
            p.requires_grad = True


# ── Vision Encoder (CNN spatial extractor) ────────────────────────────────────

class VisionEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: Optional[int] = None,
        pretrained: bool = False,
        freeze_backbone: bool = True,
        partial_unfreeze: bool = False,
        backbone_name: BackboneName = "efficientnet_b0",
    ):
        super().__init__()
        self.backbone, c_out = _build_backbone(backbone_name, pretrained)
        self.backbone_name = backbone_name

        # Freeze all first, then selectively re-enable if partial_unfreeze
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if freeze_backbone and partial_unfreeze:
            _partial_unfreeze(self.backbone, backbone_name)

        if embed_dim is None:
            self.proj = None
            self.embed_dim = c_out
        else:
            self.proj = nn.Conv2d(c_out, embed_dim, kernel_size=1)
            self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # Only skip grad computation when backbone is fully frozen
        has_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if has_trainable:
            feat = self.backbone(x_flat)
        else:
            with torch.no_grad():
                feat = self.backbone(x_flat)             # (B*T, c_out, 7, 7)

        if self.proj is not None:
            feat = self.proj(feat)

        _, _, Hf, Wf = feat.shape
        feat = feat.flatten(2).permute(0, 2, 1).contiguous()   # (B*T, N, E)
        return feat.view(B, T, Hf * Wf, feat.shape[-1])        # (B, T, N, E)


# ── Shared transformer builder ────────────────────────────────────────────────

def _make_layers(
    input_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    drop_path_rate: float,
) -> nn.ModuleList:
    """Build transformer layers with linearly increasing stochastic depth rate."""
    dpr = torch.linspace(0, drop_path_rate, num_layers).tolist()
    return nn.ModuleList([
        DropPathTransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN: stable training
            drop_path_rate=dpr[i],
        )
        for i in range(num_layers)
    ])


# ── Spatial Transformer ────────────────────────────────────────────────────────

class VisualTransformer(nn.Module):
    """Attends over spatial patch tokens within each frame."""

    def __init__(
        self,
        input_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_patches: int = 49,        # 7×7 default from 224×224 input
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, input_dim))
        self.dropout      = nn.Dropout(dropout)
        self.layers       = _make_layers(input_dim, num_layers, num_heads, dropout, drop_path_rate)
        self.norm         = nn.LayerNorm(input_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _interpolate_pos_embedding(self, num_patches: int) -> torch.Tensor:
        N = self.pos_embedding.size(1) - 1
        if N == num_patches:
            return self.pos_embedding
        cls_pe    = self.pos_embedding[:, :1]
        patch_pe  = self.pos_embedding[:, 1:].reshape(1, int(N**0.5), int(N**0.5), -1).permute(0, 3, 1, 2)
        new_side  = int(num_patches ** 0.5)
        patch_pe  = F.interpolate(patch_pe, size=(new_side, new_side), mode="bilinear", align_corners=False)
        patch_pe  = patch_pe.permute(0, 2, 3, 1).reshape(1, num_patches, -1)
        return torch.cat([cls_pe, patch_pe], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, N, D)
        BT, num_patches, _ = x.shape
        pos_emb = self._interpolate_pos_embedding(num_patches)

        cls_tokens = self.cls_token.expand(BT, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)            # (BT, N+1, D)
        x = x + pos_emb
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)[:, 0]                         # cls token → (BT, D)


# ── Temporal Transformer ──────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """Attends over frame-level embeddings across time."""

    def __init__(
        self,
        input_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 32,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.cls_token        = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.temp_pos_embedding = nn.Parameter(torch.empty(1, max_frames + 1, input_dim))
        self.dropout          = nn.Dropout(dropout)
        self.layers           = _make_layers(input_dim, num_layers, num_heads, dropout, drop_path_rate)
        self.norm             = nn.LayerNorm(input_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.temp_pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, _ = x.shape
        T1 = T + 1
        if T1 <= self.temp_pos_embedding.size(1):
            pos_emb = self.temp_pos_embedding[:, :T1]
        else:
            # Interpolate if video is longer than max_frames
            pos_emb = F.interpolate(
                self.temp_pos_embedding.permute(0, 2, 1),
                size=T1, mode="linear", align_corners=False,
            ).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)            # (B, T+1, D)
        x = x + pos_emb
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)[:, 0]                         # cls token → (B, D)
