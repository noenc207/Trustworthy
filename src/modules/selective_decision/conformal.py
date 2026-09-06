from __future__ import annotations
import numpy as np

class ConformalPredictor:
    """Split conformal prediction with marginal coverage guarantee.
    
    Provides a prediction SET rather than a single class.
    Coverage guarantee: P(y ∈ C(x)) >= 1 - alpha under exchangeability.
    
    IMPORTANT: Does NOT guarantee patient-level correctness.
    Only provides marginal coverage under exchangeability assumptions.
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_hat = None
    
    def calibrate(self, cal_softmax: np.ndarray, cal_labels: np.ndarray) -> None:
        """Fit on calibration holdout (NOT validation set used for model selection)."""
        n = len(cal_labels)
        # Nonconformity scores: 1 - softmax prob of true class
        scores = 1 - cal_softmax[np.arange(n), cal_labels]
        
        # Calculate quantile
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = np.quantile(scores, q_level, method='higher')
    
    def predict_set(self, softmax_probs: np.ndarray) -> list[set[int]]:
        """Return prediction sets for a batch of inputs."""
        if self.q_hat is None:
            raise ValueError("ConformalPredictor must be calibrated before predict_set.")
            
        n_samples, n_classes = softmax_probs.shape
        prediction_sets = []
        
        for i in range(n_samples):
            # Include classes where nonconformity score <= q_hat
            # i.e., 1 - prob <= q_hat  => prob >= 1 - q_hat
            included_classes = set(np.where(softmax_probs[i] >= 1 - self.q_hat)[0].tolist())
            if not included_classes:
                # Fallback to argmax if set is empty (rare but possible depending on q_hat)
                included_classes = {int(np.argmax(softmax_probs[i]))}
            prediction_sets.append(included_classes)
            
        return prediction_sets
    
    def coverage(self, softmax_probs: np.ndarray, true_labels: np.ndarray) -> float:
        """Compute empirical coverage on a dataset."""
        pred_sets = self.predict_set(softmax_probs)
        covered = sum(1 for i, s in enumerate(pred_sets) if true_labels[i] in s)
        return covered / len(true_labels)
