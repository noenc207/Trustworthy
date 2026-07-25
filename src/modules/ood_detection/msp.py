from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import BaseOODDetector


class MSPDetector(BaseOODDetector):
    """Maximum Softmax Probability (MSP) Detector."""

    def __init__(self, model: nn.Module, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.threshold = threshold

    def get_method_name(self) -> str:
        return "MSP"

    def get_threshold(self) -> float:
        return self.threshold

    def compute_score(self, x: torch.Tensor) -> float:
        """Score is 1.0 - max_prob. Higher means more OOD."""
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
        return float(1.0 - max_probs.mean().item())
