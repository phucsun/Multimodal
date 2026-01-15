import torch
import torch.nn as nn
from typing import Optional
from .blocks import VisionEncoder, VisualTransformer, TemporalTransformer

class FullVisionBranch(nn.Module):
    def __init__(self, embed_dim: Optional[int] = None, num_frames: int = 16, pretrained_backbone: bool = False, freeze_backbone: bool = True):
        super().__init__()
        self.spatial_cnn = VisionEncoder(embed_dim=embed_dim, pretrained=pretrained_backbone, freeze_backbone=freeze_backbone)
        D = self.spatial_cnn.embed_dim
        self.spatial_vit = VisualTransformer(input_dim=D)
        self.temporal_vit = TemporalTransformer(input_dim=D, max_frames=num_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        feat = self.spatial_cnn(x)            # (B, T, N, D)
        B, T, N, D = feat.shape
        feat_flat = feat.view(B * T, N, D).contiguous()
        spatial_emb = self.spatial_vit(feat_flat)     # (B*T, D)
        spatial_emb = spatial_emb.view(B, T, D).contiguous()
        video_embedding = self.temporal_vit(spatial_emb)  # (B, D)
        return video_embedding

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 512, num_frames: int = 16, pretrained_backbone: bool = False, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = FullVisionBranch(embed_dim=embed_dim, num_frames=num_frames, pretrained_backbone=pretrained_backbone, freeze_backbone=freeze_backbone)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(embed_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(x)
        logits = self.classifier(embedding)
        return logits