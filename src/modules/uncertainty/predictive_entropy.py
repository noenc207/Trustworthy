import numpy as np
from .base import BaseUncertaintyEstimator

class PredictiveEntropy(BaseUncertaintyEstimator):
    def estimate(self, probabilities: np.ndarray) -> dict:
        # probabilities shape (N, C)
        eps = 1e-15
        probs = np.clip(probabilities, eps, 1 - eps)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return {"entropy": float(np.mean(entropy)), "entropy_array": entropy}
