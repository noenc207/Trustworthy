from typing import Any

import numpy as np

from src.modules.explainability.strategies.base import BaseCAMStrategy


class HiResCAMStrategy(BaseCAMStrategy):
    """HiResCAM Implementation.

    Responsibility: Computes high-resolution class activation maps by element-wise multiplication of activations and gradients.
    Time Complexity: O(C * H * W) where C is channels, H is height, W is width.
    Determinism: Fully deterministic.
    Mathematical Formula: L_{HiResCAM}^c = sum_k (A^k * dy^c / dA^k).
    Edge Cases: Handles spatial gradients directly without pooling. Retains high resolution since gradients aren't averaged spatially.
    """

    def compute(self, activations: Any, gradients: Any) -> np.ndarray:
        """Compute the HiResCAM heatmap.

        Args:
            activations (Any): The feature map activations (B, C, H, W).
            gradients (Any): The gradients of the target class w.r.t the activations (B, C, H, W).

        Returns:
            np.ndarray: The raw HiResCAM heatmap (H, W).
        """
        if hasattr(activations, "detach"):
            act = activations.detach().cpu().numpy()
            grad = gradients.detach().cpu().numpy()
        else:
            act = np.array(activations)
            grad = np.array(gradients)

        cam = np.sum(grad * act, axis=1)

        if cam.shape[0] == 1:
            cam = cam[0]

        self.raw_heatmap = cam
        return cam
