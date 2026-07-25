from typing import Any

import numpy as np

from src.modules.explainability.strategies.base import BaseCAMStrategy


class GradCAMStrategy(BaseCAMStrategy):
    """GradCAM Implementation.

    Responsibility: Computes class-discriminative localization maps using gradient information.
    Time Complexity: O(C * H * W) where C is channels, H is height, W is width.
    Determinism: Fully deterministic.
    Mathematical Formula: L_{Grad-CAM}^c = ReLU(sum_k alpha_k^c A^k) where alpha_k^c = 1/Z sum_i sum_j (dy^c / dA_{i,j}^k).
    Edge Cases: Handles single-batch images correctly. Extracts the first batch item if batch size is 1.
    """

    def compute(self, activations: Any, gradients: Any) -> np.ndarray:
        """Compute the GradCAM heatmap.

        Args:
            activations (Any): The feature map activations (B, C, H, W).
            gradients (Any): The gradients of the target class w.r.t the activations (B, C, H, W).

        Returns:
            np.ndarray: The raw GradCAM heatmap (H, W).
        """
        if hasattr(activations, "detach"):
            act = activations.detach().cpu().numpy()
            grad = gradients.detach().cpu().numpy()
        else:
            act = np.array(activations)
            grad = np.array(gradients)

        weights = np.mean(grad, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * act, axis=1)

        if cam.shape[0] == 1:
            cam = cam[0]

        self.raw_heatmap = cam
        return cam
