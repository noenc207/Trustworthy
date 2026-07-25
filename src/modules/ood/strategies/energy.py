import numpy as np
from typing import Any
from src.modules.ood.strategies.base import AbstractOODStrategy
from src.modules.classifier.result import PredictionResult

class EnergyStrategy(AbstractOODStrategy):
    """Energy-based OOD Detection."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        
    def compute_score(self, classification_result: PredictionResult, raw_logits: Any = None) -> float:
        if raw_logits is None:
            # Fallback to probabilities if logits are not provided
            probs = np.array(list(classification_result.probabilities.values()))
            # This is an approximation since true energy requires logits
            energy = -self.temperature * np.log(np.sum(np.exp(probs / self.temperature)))
        else:
            # Assumes raw_logits is a numpy array
            logits = np.array(raw_logits)
            energy = -self.temperature * np.log(np.sum(np.exp(logits / self.temperature)))
            
        # Higher energy = more OOD
        return float(energy)
        
    def is_ood(self, score: float, threshold: float) -> bool:
        return score > threshold
        
    def threshold(self) -> float:
        return 10.0 # Arbitrary default for Energy
        
    def metadata(self) -> dict:
        return {
            "name": "Energy",
            "description": "Energy-based Score"
        }
