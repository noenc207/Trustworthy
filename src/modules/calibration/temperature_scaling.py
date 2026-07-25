import numpy as np
from scipy.optimize import minimize
import pickle
from .base import BaseCalibrator

class TemperatureScaling(BaseCalibrator):
    def __init__(self):
        super().__init__()
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        self.metadata["num_samples"] = len(logits)
        self.metadata["fit_dataset"] = "validation"
        
        def nll(temp):
            scaled_logits = logits / temp[0]
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            return -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            
        res = minimize(nll, [1.0], bounds=[(0.01, 100.0)])
        self.temperature = res.x[0]

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        max_logits = np.max(scaled, axis=1, keepdims=True)
        exp_logits = np.exp(scaled - max_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.temperature, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.temperature = pickle.load(f)
