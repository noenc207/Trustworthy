import cv2
import numpy as np

from src.modules.explainability.config import ExplainabilityConfig


class OverlayEngine:
    """Handles clinical-grade heatmap blending."""

    @staticmethod
    def render(
        heatmap: np.ndarray,
        original_image: np.ndarray,
        config: ExplainabilityConfig
    ) -> np.ndarray:
        if original_image.dtype != np.uint8:
            if original_image.max() <= 1.0:
                img_bgr = (original_image * 255).astype(np.uint8)
            else:
                img_bgr = original_image.astype(np.uint8)
        else:
            img_bgr = original_image.copy()

        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        h, w = img_bgr.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=config.interpolation)
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, config.colormap)

        overlay = cv2.addWeighted(img_bgr, 1.0 - config.opacity, heatmap_color, config.opacity, 0)
        return overlay
