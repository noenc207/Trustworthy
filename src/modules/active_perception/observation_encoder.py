from __future__ import annotations
import torch
import torch.nn as nn

class ObservationEncoder(nn.Module):
    """Encodes a sequence of observations into a state representation.
    
    Uses a frozen expert backbone to extract features from each observation,
    then aggregates across the observation sequence with a GRU.
    """
    def __init__(self, feature_dim: int = 1792, hidden_dim: int = 256, num_layers: int = 1, num_actions: int = 13, action_embed_dim: int = 16):
        super().__init__()
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
        self.gru = nn.GRU(hidden_dim + action_embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
    def encode_single(self, features: torch.Tensor, action_idx: int, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """Encode one observation step. Returns updated hidden state.
        
        Args:
            features: Tensor of shape (1, feature_dim) or (feature_dim,)
            action_idx: Integer index of the action taken
            h_prev: Tensor of shape (num_layers, 1, hidden_dim) representing previous hidden state
            
        Returns:
            Updated hidden state h_next of shape (num_layers, 1, hidden_dim)
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        proj_feat = self.projection(features) # (1, hidden_dim)
        action_tensor = torch.tensor([action_idx], device=features.device)
        action_emb = self.action_embedding(action_tensor) # (1, action_embed_dim)
        
        gru_input = torch.cat([proj_feat, action_emb], dim=-1).unsqueeze(1) # (1, 1, hidden_dim + action_embed_dim)
        
        _, h_next = self.gru(gru_input, h_prev)
        return h_next
        
    def encode_sequence(self, feature_sequence: list[torch.Tensor], action_sequence: list[int]) -> torch.Tensor:
        """Encode full sequence of observations. Returns final hidden state."""
        h_state = None
        for feat, act in zip(feature_sequence, action_sequence):
            h_state = self.encode_single(feat, act, h_state)
        if h_state is None:
            raise ValueError("Empty sequence provided to encode_sequence.")
        return h_state
