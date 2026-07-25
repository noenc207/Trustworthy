from typing import Any

import numpy as np

from src.modules.explainability.strategies.base import BaseCAMStrategy


class GradCAMPPStrategy(BaseCAMStrategy):
    """GradCAM++ Implementation.

    Responsibility: Computes localization maps by weighting positive gradients, capturing multiple object occurrences better.
    Time Complexity: O(C * H * W) where C is channels, H is height, W is width.
    Determinism: Fully deterministic.
    Mathematical Formula: alpha_{i,j}^{k,c} = (d^2y^c / (dA_{i,j}^k)^2) / (2 * d^2y^c / (dA_{i,j}^k)^2 + sum_a sum_b A_{a,b}^k d^3y^c / (dA_{a,b}^k)^3).
    Edge Cases: Prevents division by zero using a small epsilon (1e-8). Only positive gradients are used in weighting.
    """

    def compute(self, activations: Any, gradients: Any) -> np.ndarray:
        """Compute the GradCAM++ heatmap.

        Args:
            activations (Any): The feature map activations (B, C, H, W).
            gradients (Any): The gradients of the target class w.r.t the activations (B, C, H, W).

        Returns:
            np.ndarray: The raw GradCAM++ heatmap (H, W).
        """
        if hasattr(activations, "detach"):
            act = activations.detach().cpu().numpy()
            grad = gradients.detach().cpu().numpy()
        else:
            act = np.array(activations)
            grad = np.array(gradients)

        grad_2 = grad ** 2
        grad_3 = grad ** 3

        sum_a = np.sum(act, axis=(2, 3), keepdims=True)
        denom = 2 * grad_2 + grad_3 * sum_a
        denom = np.where(denom != 0, denom, 1e-8)

        alpha = grad_2 / denom

        weights = np.sum(np.maximum(grad, 0) * alpha, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * act, axis=1)

        if cam.shape[0] == 1:
            cam = cam[0]

        self.raw_heatmap = cam
        return cam
