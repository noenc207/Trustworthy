from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertFusion(nn.Module):
    """Fuses expert outputs using dynamic weights."""
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, expert_logits: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """Weighted fusion of expert logits. Returns fused logits (B, num_classes)."""
        stacked_logits = torch.stack(expert_logits, dim=1) # (B, num_experts, num_classes)
        weights = weights.unsqueeze(-1) # (B, num_experts, 1)
        fused_logits = (stacked_logits * weights).sum(dim=1) # (B, num_classes)
        return fused_logits
    
    def forward_with_diversity(self, expert_logits: list[torch.Tensor], weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fusion + diversity metrics (disagreement, KL between experts)."""
        fused_logits = self.forward(expert_logits, weights)
        
        stacked_probs = torch.softmax(torch.stack(expert_logits, dim=1), dim=-1) # (B, E, C)
        mean_probs = stacked_probs.mean(dim=1, keepdim=True) # (B, 1, C)
        
        disagreement = ((stacked_probs - mean_probs) ** 2).mean(dim=(1, 2)) # (B,)
        
        # approximate KL div
        kl_divs = F.kl_div(mean_probs.log().expand_as(stacked_probs).transpose(-1, -2), 
                           stacked_probs.transpose(-1, -2), reduction='none').sum(dim=-1).mean(dim=1) # (B,)

        metrics = {
            "disagreement": disagreement,
            "kl_divergence": kl_divs
        }
        return fused_logits, metrics
