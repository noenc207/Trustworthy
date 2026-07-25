import numpy as np
from scipy.optimize import minimize
import pickle
from .base import BaseCalibrator

class VectorScaling(BaseCalibrator):
    def __init__(self):
        super().__init__()
        self.weights = None
        self.bias = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        self.metadata["num_samples"] = len(logits)
        self.metadata["fit_dataset"] = "validation"
        num_classes = logits.shape[1]
        
        def nll(params):
            w = params[:num_classes]
            b = params[num_classes:]
            scaled = logits * w + b
            max_logits = np.max(scaled, axis=1, keepdims=True)
            exp_logits = np.exp(scaled - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            return -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            
        init_params = np.concatenate([np.ones(num_classes), np.zeros(num_classes)])
        res = minimize(nll, init_params)
        self.weights = res.x[:num_classes]
        self.bias = res.x[num_classes:]

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not fitted")
        scaled = logits * self.weights + self.bias
        max_logits = np.max(scaled, axis=1, keepdims=True)
        exp_logits = np.exp(scaled - max_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump((self.weights, self.bias), f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.weights, self.bias = pickle.load(f)
