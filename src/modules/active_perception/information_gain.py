from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Callable
from .action_space import ObservationAction
from .view_generator import ViewGenerator

def compute_observation_eig(
    model: nn.Module,
    image: np.ndarray,
    current_entropy: float,
    action: ObservationAction,
    view_generator: ViewGenerator,
    preprocess_fn: Callable,
    device: str = "cpu",
) -> float:
    """Compute Expected Information Gain for a single action.
    
    EIG(a|x) = H[p(y|x)] - H[p(y|x, o_a)]
    where o_a is the observation from action a.
    """
    try:
        view = view_generator.generate(image, action)
    except ValueError:
        return 0.0
        
    tensor_view = preprocess_fn(view).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(tensor_view)
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate entropy of the new prediction
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        new_entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1).item()
        
    return current_entropy - new_entropy

def compute_oracle_actions(
    model: nn.Module,
    image: np.ndarray,
    view_generator: ViewGenerator,
    preprocess_fn: Callable,
    available_actions: list[ObservationAction],
    device: str = "cpu",
) -> list[tuple[ObservationAction, float]]:
    """Compute EIG for all available actions. Returns sorted list (best first)."""
    try:
        base_view = view_generator.generate(image, ObservationAction.KEEP_FULL)
        tensor_base = preprocess_fn(base_view).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            base_logits = model(tensor_base)
            base_probs = torch.softmax(base_logits, dim=-1)
            eps = 1e-10
            current_entropy = -torch.sum(base_probs * torch.log(base_probs + eps), dim=-1).item()
    except Exception:
        current_entropy = 0.0
        
    results = []
    for action in available_actions:
        if action in (ObservationAction.STOP, ObservationAction.ABSTAIN):
            continue
        eig = compute_observation_eig(model, image, current_entropy, action, view_generator, preprocess_fn, device)
        results.append((action, eig))
        
    # Sort descending by EIG
    results.sort(key=lambda x: x[1], reverse=True)
    return results
