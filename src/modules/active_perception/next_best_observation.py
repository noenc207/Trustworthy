from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class NextBestObservationPolicy(nn.Module):
    """Policy network: maps evidence state to action distribution.
    
    Trained via imitation learning from oracle (not RL initially).
    """
    def __init__(self, state_dim: int, num_actions: int = 13, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, state: torch.Tensor, valid_action_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns action logits (masked if mask provided)."""
        logits = self.net(state)
        if valid_action_mask is not None:
            # Set logits of invalid actions to a very large negative number
            logits = logits.masked_fill(~valid_action_mask.bool(), -1e9)
        return logits
        
    def select_action(self, state: torch.Tensor, valid_action_mask: torch.Tensor, temperature: float = 1.0) -> int:
        """Sample action from policy. Greedy if temperature=0."""
        logits = self.forward(state, valid_action_mask)
        
        if temperature == 0.0:
            return int(torch.argmax(logits, dim=-1).item())
            
        probs = F.softmax(logits / temperature, dim=-1)
        action = torch.multinomial(probs, num_samples=1)
        return int(action.item())
