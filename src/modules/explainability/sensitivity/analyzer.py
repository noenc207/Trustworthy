from collections.abc import Callable

import numpy as np


class SensitivityAnalyzer:
    def __init__(self):
        self.num_perturbations = 10
        self.noise_level = 0.01

    def compute_robustness(self, image: np.ndarray, model_predict: Callable, perturbations: int = 10) -> float:
        try:
            baseline = model_predict(image)
        except Exception:
            return 0.0

        deviations = []
        for _ in range(perturbations):
            noise = np.random.normal(0, 0.01, image.shape)
            try:
                pred = model_predict(image + noise)
                # Compute difference
                deviations.append(float(np.sum(np.abs(baseline - pred))))
            except Exception:
                deviations.append(1.0)

        return max(0.0, 1.0 - np.mean(deviations))
