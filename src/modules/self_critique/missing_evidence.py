from __future__ import annotations
import torch
import torch.nn as nn

class MissingEvidenceDetector(nn.Module):
    """Identifies which evidence is missing/insufficient for confident decision.
    
    Uses both the feature representation and observation history to determine
    what additional evidence would be most informative.
    """
    def __init__(self, feature_dim: int, state_dim: int, num_evidence_types: int = 8):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(feature_dim + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_evidence_types)
        )
        
    def forward(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Returns missing evidence scores (B, num_evidence_types)."""
        combined = torch.cat([features, state], dim=1)
        logits = self.detector(combined)
        return torch.sigmoid(logits)
