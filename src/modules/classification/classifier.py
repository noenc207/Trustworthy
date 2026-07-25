"""
Skin lesion multi-class classifier.

Supports multiple backbone architectures via the `timm` library.
Key features:
  - Transfer learning from ImageNet pretrained weights
  - Label smoothing loss
  - Class-weighted sampling for imbalanced datasets
  - Test-Time Augmentation (TTA)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import timm


@dataclass
class ClassificationOutput:
    """Structured output from the classifier."""
    logits: torch.Tensor           # Raw model outputs (B, num_classes)
    probabilities: torch.Tensor    # Softmax probabilities (B, num_classes)
    predicted_class: int           # Argmax class index
    confidence: float              # Max probability score
    class_labels: list[str]        # Ordered class label names


class SkinLesionClassifier(nn.Module):
    """
    Multi-class skin lesion classifier.

    Args:
        backbone: timm model name (e.g. 'efficientnet_b4')
        num_classes: number of lesion classes
        pretrained: load ImageNet weights
        drop_rate: dropout rate before final classifier head
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        num_classes: int = 7,
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Load backbone via timm — automatic feature extraction
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,           # Remove classifier head
            drop_rate=drop_rate,
        )
        feature_dim = self.backbone.num_features

        # Custom classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if self._needs_pooling() else nn.Identity(),
            nn.Flatten(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=drop_rate),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=drop_rate / 2),
            nn.Linear(512, num_classes),
        )

    def _needs_pooling(self) -> bool:
        """Check if backbone output needs spatial pooling."""
        return "efficientnet" in self.backbone_name or "resnet" in self.backbone_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        features = self.backbone(x)
        return self.head(features)

    def predict(self, x: torch.Tensor, class_labels: list[str]) -> ClassificationOutput:
        """Run inference and return structured output."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            pred_class = int(probs.argmax(dim=-1).item())
            confidence = float(probs.max().item())

        return ClassificationOutput(
            logits=logits,
            probabilities=probs,
            predicted_class=pred_class,
            confidence=confidence,
            class_labels=class_labels,
        )
