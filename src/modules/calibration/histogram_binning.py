import numpy as np
import pickle
from .base import BaseCalibrator

class HistogramBinning(BaseCalibrator):
    def __init__(self, bins=15):
        super().__init__()
        self.bins = bins
        self.bin_edges = None
        self.bin_values = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        self.metadata["num_samples"] = len(logits)
        self.metadata["fit_dataset"] = "validation"
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        num_classes = probs.shape[1]
        self.bin_edges = np.linspace(0, 1, self.bins + 1)
        self.bin_values = np.zeros((num_classes, self.bins))
        
        for c in range(num_classes):
            p_c = probs[:, c]
            y_c = (labels == c).astype(float)
            for i in range(self.bins):
                mask = (p_c > self.bin_edges[i]) & (p_c <= self.bin_edges[i+1])
                if i == 0:
                    mask = (p_c >= self.bin_edges[i]) & (p_c <= self.bin_edges[i+1])
                if mask.any():
                    self.bin_values[c, i] = y_c[mask].mean()
                else:
                    self.bin_values[c, i] = (self.bin_edges[i] + self.bin_edges[i+1]) / 2.0

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.bin_values is None:
            raise ValueError("Model not fitted")
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        calibrated = np.zeros_like(probs)
        for c in range(probs.shape[1]):
            indices = np.digitize(probs[:, c], self.bin_edges[1:-1])
            calibrated[:, c] = self.bin_values[c, indices]
            
        calibrated = np.clip(calibrated, 1e-15, 1.0)
        return calibrated / np.sum(calibrated, axis=1, keepdims=True)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump((self.bin_edges, self.bin_values), f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.bin_edges, self.bin_values = pickle.load(f)
