from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import BaseOODDetector


class ODINDetector(BaseOODDetector):
    """
    ODIN (Out-of-DIstribution detector for Neural networks)
    Uses temperature scaling and input perturbation.
    """

    def __init__(
        self,
        model: nn.Module,
        temperature: float = 1000.0,
        epsilon: float = 0.0014,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.temperature = temperature
        self.epsilon = epsilon
        self.threshold = threshold

    def get_method_name(self) -> str:
        return "ODIN"

    def get_threshold(self) -> float:
        return self.threshold

    def compute_score(self, x: torch.Tensor) -> float:
        """ODIN Score. Higher means more OOD."""

        # Ensure we don't pollute global gradients
        with torch.enable_grad():
            x_clone = x.clone().detach().requires_grad_(True)

            logits = self.model(x_clone)
            logits_scaled = logits / self.temperature

            # Maximize log probability of the predicted class
            max_logits, _ = torch.max(logits_scaled, dim=-1)
            loss = torch.sum(max_logits)
            loss.backward()

            gradient = x_clone.grad.data
            gradient_sign = torch.sign(gradient)

        # Perturb the input
        x_perturbed = x_clone - self.epsilon * gradient_sign

        # Forward pass on perturbed image
        with torch.no_grad():
            logits_perturbed = self.model(x_perturbed)
            probs = torch.softmax(logits_perturbed / self.temperature, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]

        return float(1.0 - max_probs.mean().item())
