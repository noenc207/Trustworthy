from typing import Any

import numpy as np

from src.modules.classifier.result import PredictionResult
from src.modules.ood.strategies.base import AbstractOODStrategy


class EntropyStrategy(AbstractOODStrategy):
    """Predictive Entropy OOD Detection."""

    def compute_score(self, classification_result: PredictionResult, raw_logits: Any = None) -> float:
        probs = np.array(list(classification_result.probabilities.values()))
        # Add epsilon to prevent log(0)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        return float(entropy)

    def is_ood(self, score: float, threshold: float) -> bool:
        return score > threshold

    def threshold(self) -> float:
        return 1.5 # Arbitrary default for Entropy

    def metadata(self) -> dict:
        return {
            "name": "Entropy",
            "description": "Shannon Entropy of predictive distribution"
        }
