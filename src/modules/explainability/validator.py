from typing import Any

import numpy as np

from src.modules.explainability.exceptions import ExplanationValidationError


class ExplanationValidator:
    """Validates intermediate and final heatmaps to prevent silent pipeline corruption."""

    @staticmethod
    def validate_tensors(activations: Any, gradients: Any) -> None:
        if activations is None or gradients is None:
            raise ExplanationValidationError("Activations or gradients are missing. Hook failure.")

        import torch
        if isinstance(activations, torch.Tensor):
            acts = activations.detach().cpu().numpy()
        else:
            acts = np.array(activations)

        if isinstance(gradients, torch.Tensor):
            grads = gradients.detach().cpu().numpy()
        else:
            grads = np.array(gradients)

        if np.isnan(acts).any() or np.isnan(grads).any():
            raise ExplanationValidationError("NaN detected in hooks.")

        if np.isinf(acts).any() or np.isinf(grads).any():
            raise ExplanationValidationError("Inf detected in hooks.")

        if np.var(acts) < 1e-8:
            raise ExplanationValidationError("Activation variance is near zero.")

    @staticmethod
    def validate_heatmap(heatmap: np.ndarray) -> None:
        if heatmap is None or heatmap.size == 0:
            raise ExplanationValidationError("Generated heatmap is empty.")

        if np.isnan(heatmap).any() or np.isinf(heatmap).any():
            raise ExplanationValidationError("Heatmap contains NaN or Inf.")

        if np.max(heatmap) < 1e-8:
            raise ExplanationValidationError("Heatmap is entirely zero.")

        if np.var(heatmap) < 1e-8:
            raise ExplanationValidationError("Heatmap variance is near zero.")
