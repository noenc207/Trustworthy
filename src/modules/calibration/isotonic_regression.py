import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression as SklearnIsotonic

from .base import BaseCalibrator


class IsotonicRegression(BaseCalibrator):
    def __init__(self):
        super().__init__()
        self.ir_models = []

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        self.metadata["num_samples"] = len(logits)
        self.metadata["fit_dataset"] = "validation"
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        num_classes = probs.shape[1]
        self.ir_models = []
        for c in range(num_classes):
            ir = SklearnIsotonic(out_of_bounds="clip")
            y_c = (labels == c).astype(float)
            ir.fit(probs[:, c], y_c)
            self.ir_models.append(ir)

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self.ir_models:
            raise ValueError("Model not fitted")
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        calibrated = np.zeros_like(probs)
        for c in range(probs.shape[1]):
            calibrated[:, c] = self.ir_models[c].predict(probs[:, c])

        calibrated = np.clip(calibrated, 1e-15, 1.0)
        return calibrated / np.sum(calibrated, axis=1, keepdims=True)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.ir_models, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.ir_models = pickle.load(f)
