from abc import ABC, abstractmethod
from typing import Any

from src.modules.classifier.result import PredictionResult
from src.modules.ood.result import OODResult


class OODStrategy(ABC):
    """Strategy pattern for Out-of-Distribution algorithms."""

    @abstractmethod
    def compute_score(self, classification_result: PredictionResult, raw_logits: Any = None) -> float:
        """Compute the raw OOD score."""
        pass

    @abstractmethod
    def is_ood(self, score: float, threshold: float) -> bool:
        """Determine if the score indicates OOD."""
        pass

    @abstractmethod
    def threshold(self) -> float:
        """Default recommended threshold for the algorithm."""
        pass

    @abstractmethod
    def metadata(self) -> dict:
        """Metadata about the algorithm."""
        pass

class OODDetector(ABC):
    """Orchestrator for evaluating OOD status."""

    @abstractmethod
    def evaluate(self, classification_result: PredictionResult, raw_logits: Any = None) -> OODResult:
        """Evaluate input and return formal OODResult."""
        pass
