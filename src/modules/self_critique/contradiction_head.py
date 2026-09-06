from __future__ import annotations
import torch
import torch.nn as nn

class ContradictionEvidenceHead(nn.Module):
    """Predicts which evidence types contradict the current prediction.
    
    Multi-label classification: each output dimension corresponds to
    an evidence type.
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
