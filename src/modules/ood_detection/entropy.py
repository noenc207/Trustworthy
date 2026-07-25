from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import BaseOODDetector


class EntropyDetector(BaseOODDetector):
    """Predictive Entropy OOD Detection."""

    def __init__(self, model: nn.Module, threshold: float = 1.5, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.threshold = threshold

    def get_method_name(self) -> str:
        return "Entropy"

    def get_threshold(self) -> float:
        return self.threshold

    def compute_score(self, x: torch.Tensor) -> float:
        """Shannon Entropy of the predictive distribution."""
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            # Add epsilon to prevent log(0) which causes NaN
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        return float(entropy.mean().item())
