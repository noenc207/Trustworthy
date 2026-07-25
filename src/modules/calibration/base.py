from abc import ABC, abstractmethod
import numpy as np

class BaseCalibrator(ABC):
    def __init__(self):
        self.metadata = {
            "fit_dataset": "unknown",
            "num_samples": 0,
            "random_seed": 42
        }

    @abstractmethod
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        pass

    @abstractmethod
    def transform(self, logits: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass
