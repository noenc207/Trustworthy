"""
MC Dropout implementation for uncertainty estimation.
"""
import torch
import torch.nn as nn

def enable_mc_dropout(model: nn.Module):
    """
    Enable MC Dropout by keeping the model in eval mode,
    but switching all Dropout layers to train mode.
    """
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def compute_predictive_entropy(p_mean: torch.Tensor) -> torch.Tensor:
    """Compute predictive entropy H(P_mean)."""
    # p_mean: (B, C)
    entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-8), dim=1)
    return entropy

def compute_mutual_information(probs_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute predictive entropy, mean entropy, and mutual information.
    probs_stack: (N, B, C)
    """
    p_mean = torch.mean(probs_stack, dim=0) # (B, C)
    predictive_entropy = compute_predictive_entropy(p_mean)
    
    # Entropy for each pass
    entropies = -torch.sum(probs_stack * torch.log(probs_stack + 1e-8), dim=-1) # (N, B)
    mean_entropy = torch.mean(entropies, dim=0) # (B,)
    
    mutual_information = predictive_entropy - mean_entropy
    
    return predictive_entropy, mean_entropy, mutual_information
