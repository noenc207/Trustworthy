from __future__ import annotations
from src.modules.selective_decision.risk_controller import SelectiveDecision

class AbstentionTracker:
    """Tracks abstention statistics for evaluation."""
    def __init__(self):
        self.total_samples = 0
        self.abstentions = 0
        self.reviews = 0
        self.differentials = 0
        self.classifications = 0
        self.correct_classifications = 0
        self.fn_count = 0  # Assuming FN means incorrect classifications when we shouldn't have classified
        
    def update(self, decision: SelectiveDecision, true_label: int, predicted_label: int, confidence: float) -> None:
        self.total_samples += 1
        
        if decision == SelectiveDecision.ABSTAIN:
            self.abstentions += 1
        elif decision == SelectiveDecision.REVIEW:
            self.reviews += 1
        elif decision == SelectiveDecision.DIFFERENTIAL:
            self.differentials += 1
        elif decision == SelectiveDecision.CLASSIFY:
            self.classifications += 1
            if true_label == predicted_label:
                self.correct_classifications += 1
            else:
                self.fn_count += 1
    
    def compute_metrics(self) -> dict[str, float]:
        """Returns: coverage, abstention_rate, selective_accuracy, selective_risk, accepted_fn_rate."""
        if self.total_samples == 0:
            return {}
            
        abstention_rate = self.abstentions / self.total_samples
        coverage = 1.0 - abstention_rate
        
        if self.classifications > 0:
            selective_accuracy = self.correct_classifications / self.classifications
            selective_risk = 1.0 - selective_accuracy
            accepted_fn_rate = self.fn_count / self.classifications
        else:
            selective_accuracy = 0.0
            selective_risk = 0.0
            accepted_fn_rate = 0.0
            
        return {
            "coverage": coverage,
            "abstention_rate": abstention_rate,
            "selective_accuracy": selective_accuracy,
            "selective_risk": selective_risk,
            "accepted_fn_rate": accepted_fn_rate,
            "review_rate": self.reviews / self.total_samples,
            "differential_rate": self.differentials / self.total_samples
        }
