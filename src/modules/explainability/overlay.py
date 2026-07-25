
import cv2
import numpy as np

from src.modules.explainability.config import ExplainabilityConfig


class OverlayEngine:
    """Handles blending, colormaps, and resizing for heatmaps."""

    @staticmethod
    def render(
        heatmap: np.ndarray,
        original_image: np.ndarray,
        config: ExplainabilityConfig,
        bounding_box: tuple[int, int, int, int] | None = None
    ) -> np.ndarray:
        """Renders the heatmap over the original image."""

        # 1. Ensure original_image is uint8 BGR
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

        # 2. Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=config.interpolation)

        # 3. Apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, config.colormap)

        # 4. Blend
        overlay = cv2.addWeighted(img_bgr, 1.0 - config.opacity, heatmap_color, config.opacity, 0)

        # 5. Draw Bounding Box if provided
        if bounding_box:
            x, y, bw, bh = bounding_box
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        return overlay
