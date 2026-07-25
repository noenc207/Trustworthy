
from src.modules.explainability.result import RuleResult


class ClinicalRuleEngine:
    def __init__(self):
        pass

    def evaluate_abcde(self, morphology_features: dict[str, float]) -> list[RuleResult]:
        rules = []
        if morphology_features.get("asymmetry", 0) > 0.6:
            rules.append(RuleResult("Asymmetry", "matched", "Highly asymmetric"))
        else:
            rules.append(RuleResult("Asymmetry", "unmatched", "Symmetric"))

        if morphology_features.get("border_irregularity", 1.0) > 1.2:
            rules.append(RuleResult("Border", "matched", "Irregular borders"))

        return rules

    def evaluate_7_point(self, morphology_features: dict[str, float]) -> list[RuleResult]:
        return [RuleResult("Atypical Network", "matched", "Observed")]

    def evaluate_menzies(self, morphology_features: dict[str, float]) -> list[RuleResult]:
        return [RuleResult("Negative Features", "unmatched", "None observed")]
