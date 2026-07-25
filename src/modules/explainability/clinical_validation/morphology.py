import contextlib

import numpy as np

with contextlib.suppress(ImportError):
    import cv2


class MorphologicalEngine:
    def __init__(self):
        self.enabled_features = ["asymmetry", "border", "color"]
        self.resolution = 224

    def compute_morphology(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        asymmetry_score = 0.5
        border_irregularity = 0.5
        color_var = 0.5
        diameter = 0.5

        if mask is not None and len(mask.shape) >= 2:
            try:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = contours[0]
                    perimeter = cv2.arcLength(cnt, True)
                    area = cv2.contourArea(cnt)
                    if area > 0:
                        border_irregularity = (perimeter ** 2) / (4 * np.pi * area)
            except Exception:
                pass

        return {
            "asymmetry": asymmetry_score,
            "border_irregularity": border_irregularity,
            "color_variation": color_var,
            "diameter": diameter
        }
