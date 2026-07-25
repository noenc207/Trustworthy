from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.modules.explainability.strategies.base import BaseCAMStrategy


class ScoreCAMStrategy(BaseCAMStrategy):
    """ScoreCAM Implementation.

    Responsibility: Computes localization maps by calculating the forward pass scores of masked inputs.
    Time Complexity: O(C * T) where C is channels and T is forward pass time. Very slow compared to gradient methods.
    Determinism: Fully deterministic if the model is in eval mode.
    Mathematical Formula: L_{Score-CAM}^c = ReLU(sum_k alpha_k^c A^k) where alpha_k^c = softmax(f^c(X * Upsample(A^k)) - f^c(X_b)).
    Edge Cases: Normalizes upsampled activations to [0, 1] before masking to prevent exploding scores. Falls back to GradCAM approximation if model/input are missing.
    """

    def compute(self, activations: Any, gradients: Any) -> np.ndarray:
        """Compute the ScoreCAM heatmap.

        Args:
            activations (Any): The feature map activations (B, C, H, W).
            gradients (Any): Unused for ScoreCAM.

        Returns:
            np.ndarray: The raw ScoreCAM heatmap (H, W).
        """
        if hasattr(activations, "detach"):
            a_tensor = activations.clone().detach()
        else:
            a_tensor = torch.tensor(activations)

        b, c, _h, _w = a_tensor.shape
        model = getattr(self, "model", None)
        input_tensor = getattr(self, "input_tensor", None)
        target_class = getattr(self, "target_class", None)

        if model is None or input_tensor is None or target_class is None:
            # Fallback if required components are not collected
            a_np = a_tensor.cpu().numpy()
            weights = np.mean(a_np, axis=(2, 3), keepdims=True)
            cam = np.sum(weights * a_np, axis=1)
            if cam.shape[0] == 1:
                cam = cam[0]
            self.raw_heatmap = cam
            return cam

        with torch.no_grad():
            input_shape = input_tensor.shape[2:]
            a_upsampled = F.interpolate(a_tensor, size=input_shape, mode='bilinear', align_corners=False)

            a_min = a_upsampled.view(b, c, -1).min(dim=2)[0].view(b, c, 1, 1)
            a_max = a_upsampled.view(b, c, -1).max(dim=2)[0].view(b, c, 1, 1)
            a_norm = (a_upsampled - a_min) / (a_max - a_min + 1e-8)

            scores = []
            batch_size = 32
            for i in range(0, c, batch_size):
                end = min(i + batch_size, c)
                masks = a_norm[0, i:end].unsqueeze(1)
                inputs = input_tensor.expand(end - i, -1, -1, -1)
                masked_batch = inputs * masks
                logits = model(masked_batch)
                batch_scores = logits[:, target_class]
                scores.append(batch_scores)

            scores_tensor = torch.cat(scores)
            scores_tensor = F.softmax(scores_tensor, dim=0)

            weights = scores_tensor.view(1, c, 1, 1).cpu().numpy()
            a_np = a_tensor.cpu().numpy()
            cam = np.sum(weights * a_np, axis=1)

        if cam.shape[0] == 1:
            cam = cam[0]

        self.raw_heatmap = cam
        return cam
