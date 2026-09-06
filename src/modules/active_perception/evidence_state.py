from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class EvidenceState:
    """Complete state representation at step t of the active perception loop."""
    h_t: torch.Tensor              # Learned state from GRU (hidden_dim,)
    p_t: torch.Tensor              # Current posterior distribution (num_classes,)
    u_epistemic: float             # Epistemic uncertainty (from MC Dropout)
    u_aleatoric: float             # Aleatoric uncertainty
    ood_score: float               # Energy-based OOD score
    quality_score: float           # Image quality score
    evidence_support: torch.Tensor # (num_evidence_dims,)
    evidence_contradiction: torch.Tensor # (num_evidence_dims,)
    evidence_missing: torch.Tensor # (num_evidence_dims,)
    observation_history: list[int] # Action indices taken so far
    budget_remaining: int
    step: int
    
    def to_tensor(self) -> torch.Tensor:
        """Flatten state into a single vector for policy network input."""
        scalars = torch.tensor([
            self.u_epistemic, 
            self.u_aleatoric, 
            self.ood_score, 
            self.quality_score,
            float(self.budget_remaining),
            float(self.step)
        ], device=self.h_t.device)
        
        return torch.cat([
            self.h_t.flatten(),
            self.p_t.flatten(),
            scalars,
            self.evidence_support.flatten(),
            self.evidence_contradiction.flatten(),
            self.evidence_missing.flatten()
        ])
    
    @staticmethod
    def state_dim(hidden_dim: int = 256, num_classes: int = 7, num_evidence_dims: int = 8) -> int:
        """Calculate total dimension of flattened state vector."""
        return hidden_dim + num_classes + 6 + 3 * num_evidence_dims
