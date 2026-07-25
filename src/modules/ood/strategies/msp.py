from typing import Any
from src.modules.ood.strategies.base import AbstractOODStrategy
from src.modules.classifier.result import PredictionResult

class MSPStrategy(AbstractOODStrategy):
    """Maximum Softmax Probability (MSP) Strategy."""
    
    def compute_score(self, classification_result: PredictionResult, raw_logits: Any = None) -> float:
        # For MSP, the score is simply the maximum probability (confidence)
        # Higher score = more In-Distribution
        # We'll use 1.0 - confidence so that higher score = more OOD
        return 1.0 - classification_result.confidence
        
    def is_ood(self, score: float, threshold: float) -> bool:
        # Score is 1 - max_prob. If 1 - max_prob > threshold, it's OOD.
        # i.e., max_prob < (1 - threshold)
        return score > threshold
        
    def threshold(self) -> float:
        # Default: if max confidence < 0.5, consider it OOD
        return 0.5
        
    def metadata(self) -> dict:
        return {
            "name": "MSP",
            "description": "Maximum Softmax Probability"
        }
