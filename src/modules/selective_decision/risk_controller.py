from __future__ import annotations
from enum import Enum

class SelectiveDecision(str, Enum):
    CLASSIFY = "classify"       # Confident single-class prediction
    DIFFERENTIAL = "differential"  # Conformal set > 1 class
    REVIEW = "review"           # High risk, needs specialist review
    ABSTAIN = "abstain"         # OOD or extremely uncertain

class RiskController:
    """Final decision logic combining all safety signals."""
    def __init__(self, ood_threshold: float, fragility_threshold: float, conformal_alpha: float = 0.1):
        self.ood_threshold = ood_threshold
        self.fragility_threshold = fragility_threshold
        self.conformal_alpha = conformal_alpha
    
    def decide(self, ood_score: float, fragility: float, conformal_set: set[int], uncertainty: float) -> SelectiveDecision:
        """Make selective decision based on all safety signals."""
        if ood_score > self.ood_threshold or uncertainty > 0.9:
            return SelectiveDecision.ABSTAIN
            
        if fragility > self.fragility_threshold:
            return SelectiveDecision.REVIEW
            
        if len(conformal_set) > 1:
            return SelectiveDecision.DIFFERENTIAL
            
        return SelectiveDecision.CLASSIFY
