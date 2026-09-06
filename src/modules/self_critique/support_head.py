from __future__ import annotations
from enum import Enum
import torch
import torch.nn as nn

class EvidenceType(Enum):
    ASYMMETRIC_BORDER = 0
    IRREGULAR_PIGMENT = 1
    VASCULAR_PATTERN = 2
    COLOR_VARIATION = 3
    STRUCTURAL_PATTERN = 4
    TEXTURE_ABNORMALITY = 5
    SIZE_INDICATOR = 6
    SHAPE_REGULARITY = 7

class SupportEvidenceHead(nn.Module):
    """Predicts which evidence types support the current prediction.
    
    Multi-label classification: each output dimension corresponds to
    an evidence type (e.g., asymmetric border, irregular pigment, etc.).
    Trained with pseudo-labels from counterfactual analysis.
    """
    def __init__(self, feature_dim: int, num_evidence_types: int = 8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_evidence_types)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid scores (B, num_evidence_types)."""
        logits = self.head(features)
        return torch.sigmoid(logits)
