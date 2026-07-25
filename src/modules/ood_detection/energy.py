from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import BaseOODDetector


class EnergyDetector(BaseOODDetector):
    """Energy-based OOD Detection."""

    def __init__(
        self,
        model: nn.Module,
        temperature: float = 1.0,
        threshold: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.temperature = temperature
        self.threshold = threshold

    def get_method_name(self) -> str:
        return "Energy"

    def get_threshold(self) -> float:
        return self.threshold

    def compute_score(self, x: torch.Tensor) -> float:
        """Negative energy. Higher score means more OOD."""
        with torch.no_grad():
            logits = self.model(x)
            # Energy = -T * logsumexp(logits / T)
            energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)
        return float(energy.mean().item())
