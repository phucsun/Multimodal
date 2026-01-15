import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

def _get_resnet18(pretrained: bool):
    try:
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        return models.resnet18(weights=weights)
    except Exception:
        return models.resnet18(pretrained=pretrained)

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim: Optional[int] = None, pretrained: bool = False, freeze_backbone: bool = True):
        super().__init__()
        resnet18 = _get_resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])
        self.c_out = 512
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if embed_dim is None:
            self.proj = None
            self.embed_dim = self.c_out
        else:
            self.proj = nn.Conv2d(self.c_out, embed_dim, kernel_size=1)
            self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feat = self.backbone(x_reshaped)  # (B*T, C_out, Hf, Wf)
        if self.proj is not None:
            feat = self.proj(feat)
        _, _, Hf, Wf = feat.shape
        N = Hf * Wf
        feat = feat.flatten(2).permute(0, 2, 1).contiguous()  # (B*T, N, E)
        feat = feat.view(B, T, N, feat.shape[-1])             # (B, T, N, E)
        return feat

class VisualTransformer(nn.Module):
    def __init__(self, input_dim=512, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = input_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embedding = None
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=self.hidden_dim*4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.hidden_dim)
        try:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        except Exception:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _init_pos_embedding(self, num_patches, device, dtype):
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, self.hidden_dim, device=device, dtype=dtype))
        try:
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        except Exception:
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, dim = x.shape
        if self.pos_embedding is None or self.pos_embedding.size(1) != (1 + num_patches):
            self._init_pos_embedding(num_patches, x.device, x.dtype)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (num_patches + 1), :]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0, :]

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=512, num_layers=4, num_heads=8, dropout=0.1, max_frames=32):
        super().__init__()
        self.hidden_dim = input_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.temp_pos_embedding = nn.Parameter(torch.randn(1, max_frames + 1, self.hidden_dim))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=self.hidden_dim*4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.hidden_dim)
        try:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.temp_pos_embedding, std=0.02)
        except Exception:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            nn.init.normal_(self.temp_pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if self.temp_pos_embedding.size(1) < (T + 1):
            new = nn.Parameter(torch.randn(1, T + 1, self.hidden_dim, device=x.device, dtype=x.dtype))
            try:
                nn.init.trunc_normal_(new, std=0.02)
            except Exception:
                nn.init.normal_(new, mean=0.0, std=0.02)
            with torch.no_grad():
                new.data[:, : self.temp_pos_embedding.size(1), :] = self.temp_pos_embedding.data
            self.temp_pos_embedding = new
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.temp_pos_embedding[:, : T + 1, :].to(x.dtype)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0, :]