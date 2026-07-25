from typing import Any


class RiskFactorEngine:
    def __init__(self):
        self.weights = {"hair": 0.2, "bubble": 0.15, "marker": 0.15, "blur": 0.3, "flash": 0.2}

    def evaluate_risk(self, artifact_confidences: dict[str, float], patient_metadata: dict[str, Any] | None = None) -> float:
        base_risk = 0.0
        weights = {
            "hair": 0.2,
            "bubble": 0.15,
            "marker": 0.15,
            "blur": 0.3,
            "flash": 0.2
        }
        for artifact, conf in artifact_confidences.items():
            base_risk += conf * weights.get(artifact, 0.0)

        if patient_metadata and patient_metadata.get("age", 0) > 80:
            base_risk += 0.05

        return min(1.0, base_risk)
