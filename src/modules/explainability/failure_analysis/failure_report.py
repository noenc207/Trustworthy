from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifact_detector import ArtifactDetector
from .failure_classifier import FailureClassifier
from .risk_factor_engine import RiskFactorEngine


@dataclass
class FailureModeResult:
    failure_probability: float
    primary_cause: str
    secondary_cause: str
    artifact_confidence: dict[str, float]
    recommended_action: str

class FailureAnalyzer:
    def __init__(self):
        self.detector = ArtifactDetector()
        self.classifier = FailureClassifier()
        self.risk_engine = RiskFactorEngine()

    def analyze(self, image: np.ndarray, patient_metadata: dict[str, Any] | None = None) -> FailureModeResult:
        artifact_confidence = self.detector.detect_all(image)
        primary_cause, secondary_cause = self.classifier.classify(artifact_confidence)
        failure_probability = self.risk_engine.evaluate_risk(artifact_confidence, patient_metadata)
        recommended_action = self.classifier.recommend_action(primary_cause)

        return FailureModeResult(
            failure_probability=failure_probability,
            primary_cause=primary_cause,
            secondary_cause=secondary_cause,
            artifact_confidence=artifact_confidence,
            recommended_action=recommended_action
        )
