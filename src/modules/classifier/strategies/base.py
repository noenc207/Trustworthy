from abc import ABC
from src.modules.classifier.interfaces import ClassifierStrategy
import numpy as np
from typing import Any

class AbstractClassifierStrategy(ClassifierStrategy, ABC):
    """Base class for all classifier backbones providing common utilities."""
    
    def _normalize_logits(self, logits: np.ndarray) -> np.ndarray:
        # Simple softmax implementation
        exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
