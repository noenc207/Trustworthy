from __future__ import annotations
import torch
import torch.nn as nn

class StoppingPolicy(nn.Module):
    """Learns when to stop acquiring evidence.
    
    Computes V_stop(s) and V_continue(s) from the evidence state.
    STOP when V_stop >= max_a V_continue(s,a) AND safety constraints met.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # output: [V_stop, V_continue]
        )
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (v_stop, v_continue) values."""
        values = self.net(state)
        v_stop = values[:, 0]
        v_continue = values[:, 1]
        return v_stop, v_continue
    
    def should_stop(self, state: torch.Tensor, safety_margin: float = 0.0) -> bool:
        """Decision: should we stop?"""
        v_stop, v_continue = self.forward(state)
        return (v_stop >= v_continue + safety_margin).item()
