from __future__ import annotations
import torch
import torch.nn as nn

class DynamicExpertRouter(nn.Module):
    """Learns input-dependent expert weights.
    
    Instead of fixed (p1+p2+p3)/3, computes:
        w_i(x) = softmax(gate(concat(z_1, ..., z_K, s)))
        z_fused = sum(w_i * z_i)
    
    where z_i are expert logits/features and s is optional state.
    """
    def __init__(self, feature_dim: int, num_experts: int = 3, state_dim: int = 0, hidden_dim: int = 256):
        super().__init__()
        self.num_experts = num_experts
        input_dim = feature_dim * num_experts + state_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, expert_features: list[torch.Tensor], state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (weights, routing_entropy). weights shape: (B, num_experts)."""
        concatenated = torch.cat(expert_features, dim=1)
        if state is not None:
            concatenated = torch.cat([concatenated, state], dim=1)
        
        logits = self.gate(concatenated)
        weights = torch.softmax(logits, dim=-1)
        
        # Calculate routing entropy
        routing_entropy = -(weights * torch.log(weights + 1e-9)).sum(dim=-1)
        
        return weights, routing_entropy
