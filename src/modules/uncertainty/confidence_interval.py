import numpy as np

from .base import BaseUncertaintyEstimator


class ConfidenceInterval(BaseUncertaintyEstimator):
    def estimate(self, probabilities: np.ndarray) -> dict:
        if probabilities.ndim == 3:
            # (M, N, C)
            mean_probs = np.mean(probabilities, axis=0)
            std_probs = np.std(probabilities, axis=0)

            # 95% CI
            lower = np.clip(mean_probs - 1.96 * std_probs, 0.0, 1.0)
            upper = np.clip(mean_probs + 1.96 * std_probs, 0.0, 1.0)

            # Max confidence width
            width = np.mean(upper - lower, axis=1)
        else:
            width = np.zeros(probabilities.shape[0])

        return {"confidence_width": float(np.mean(width)), "width_array": width}
