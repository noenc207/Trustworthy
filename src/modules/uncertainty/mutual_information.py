import numpy as np
from .base import BaseUncertaintyEstimator

class MutualInformation(BaseUncertaintyEstimator):
    def estimate(self, probabilities: np.ndarray) -> dict:
        # probabilities shape (M, N, C) where M is number of ensemble models or MC dropout samples
        if probabilities.ndim != 3:
            return {"mutual_information": 0.0, "mi_array": np.zeros(probabilities.shape[0])}
            
        eps = 1e-15
        probs = np.clip(probabilities, eps, 1 - eps)
        
        # Mean probability across models
        mean_probs = np.mean(probs, axis=0)
        
        # Predictive entropy
        total_entropy = -np.sum(mean_probs * np.log(mean_probs), axis=1)
        
        # Expected entropy
        expected_entropy = np.mean(-np.sum(probs * np.log(probs), axis=2), axis=0)
        
        mi = total_entropy - expected_entropy
        return {"mutual_information": float(np.mean(mi)), "mi_array": mi}
