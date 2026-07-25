"""Clinical ontology graph."""


class ClinicalOntology:
    """Clinical ontology graph (ABCDE rules, etc.).

    Responsibility: Represents medical knowledge and verifies feature plausibility.
    Time Complexity: O(1) for rule lookup.
    Determinism: Deterministic.
    Mathematical Formula: None directly, logic lookup function V(f, v) -> [0.0, 1.0].
    Edge Cases:
        - Unknown feature keys.
    """

    def __init__(self) -> None:
        """Initializes default ABCDE ontology."""
        self.rules: dict[str, dict[str, float]] = {
            "asymmetry": {"min": 0.0, "max": 1.0, "validity": 1.0},
            "border_irregularity": {"min": 0.0, "max": 1.0, "validity": 1.0},
            "color_variegation": {"min": 0.0, "max": 1.0, "validity": 1.0},
            "diameter": {"min": 0.0, "max": 100.0, "validity": 1.0},
            "evolution": {"min": 0.0, "max": 1.0, "validity": 1.0},
        }

    def verify(self, feature_name: str, value: float) -> float:
        """Verify feature against ontology rules.

        Args:
            feature_name: Feature key.
            value: Observed value.

        Returns:
            Validity multiplier [0, 1].
        """
        if feature_name not in self.rules:
            # Safely return a robust fallback instead of 0
            return 0.5

        rule = self.rules[feature_name]
        min_val = rule.get("min", 0.0)
        max_val = rule.get("max", 1.0)

        if min_val <= value <= max_val:
            return rule.get("validity", 1.0)
        else:
            return 0.1  # Low validity if out of expected bounds
