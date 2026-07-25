from typing import Any
from src.modules.ood.strategies.base import AbstractOODStrategy
from src.modules.classifier.result import PredictionResult

class MahalanobisStrategy(AbstractOODStrategy):
    """Mahalanobis Distance OOD Detection (Architecture Only)."""
    
    def compute_score(self, classification_result: PredictionResult, raw_logits: Any = None) -> float:
        # Requires deep feature extraction which isn't available from prediction result directly.
        # This is just an architectural stub as requested.
        return 0.0
        
    def is_ood(self, score: float, threshold: float) -> bool:
        return score > threshold
        
    def threshold(self) -> float:
        return 100.0
        
    def metadata(self) -> dict:
        return {
            "name": "Mahalanobis",
            "description": "Mahalanobis Distance in feature space"
        }
