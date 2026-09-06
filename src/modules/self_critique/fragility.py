from __future__ import annotations
import torch
from dataclasses import dataclass
from src.modules.self_critique.counterfactual_engine import CounterfactualEngine

@dataclass
class FragilityResult:
    fragility_score: float          # max KL divergence
    mean_sensitivity: float         # mean KL across regions
    most_critical_region: int       # region with highest sensitivity
    region_sensitivities: dict[int, float]  # per-region KL divergences
    is_fragile: bool                # fragility > threshold

class DecisionFragility:
    """Measures how fragile a prediction is to evidence removal.
    
    fragility = max_k S_k (maximum KL divergence across all evidence removals)
    
    High confidence + high fragility = UNSAFE prediction.
    """
    def __init__(self, counterfactual_engine: CounterfactualEngine, threshold: float = 0.5):
        self.engine = counterfactual_engine
        self.threshold = threshold
    
    def compute(self, image_tensor: torch.Tensor) -> FragilityResult:
        """Compute fragility score and identify most critical evidence."""
        sensitivities = self.engine.compute_sensitivity(image_tensor)
        
        region_kl = {k: v["kl_div"] for k, v in sensitivities.items()}
        
        max_k = max(region_kl, key=region_kl.get)
        max_kl = region_kl[max_k]
        mean_kl = sum(region_kl.values()) / len(region_kl)
        
        return FragilityResult(
            fragility_score=max_kl,
            mean_sensitivity=mean_kl,
            most_critical_region=max_k,
            region_sensitivities=region_kl,
            is_fragile=max_kl > self.threshold
        )
