"""
Inference Engine — Unified prediction pipeline.

Orchestrates the full prediction flow:
  1. Image quality check
  2. Preprocessing
  3. OOD detection
  4. Classification
  5. Uncertainty estimation
  6. Confidence calibration
  7. Grad-CAM explanation
  8. Clinical recommendation generation

Supports both PyTorch and ONNX Runtime backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from src.core.constants import LesionClass
from src.core.exceptions import ImageQualityError
from src.modules.calibration.calibrator import TemperatureScaling
from src.modules.classification.classifier import SkinLesionClassifier
from src.modules.clinical_recommendation.engine import (
    ClinicalRecommendation,
    ClinicalRecommendationEngine,
)
from src.modules.explainability.gradcam import GradCAMExplainer, GradCAMOutput
from src.modules.ood_detection import BaseOODDetector, OODResult
from src.modules.quality_assessment.assessor import ImageQualityAssessor, QualityReport
from src.modules.uncertainty.estimator import MCDropoutEstimator, UncertaintyOutput


@dataclass
class PredictionResult:
    """Complete prediction result from the inference engine."""
    # Core prediction
    predicted_class: str
    predicted_class_index: int
    probabilities: dict[str, float]     # Class name → probability
    raw_confidence: float               # Pre-calibration confidence
    calibrated_confidence: float        # Post-temperature-scaling confidence
    # Quality & Trustworthiness
    quality_report: QualityReport
    ood_result: OODResult
    uncertainty: UncertaintyOutput
    # Explainability
    gradcam: GradCAMOutput | None
    # Clinical
    recommendation: ClinicalRecommendation
    # Metadata
    model_version: str
    inference_backend: str              # 'pytorch' or 'onnx'


class TrustworthyInferenceEngine:
    """
    Central inference orchestrator for the platform.

    All components are injected, making the engine fully testable
    and swappable (dependency injection pattern).
    """

    CLASS_LABELS: ClassVar[list[str]] = [c.value for c in LesionClass]

    def __init__(
        self,
        classifier: SkinLesionClassifier,
        ood_detector: BaseOODDetector,
        uncertainty_estimator: MCDropoutEstimator,
        calibrator: TemperatureScaling,
        quality_assessor: ImageQualityAssessor,
        explainer: GradCAMExplainer,
        recommendation_engine: ClinicalRecommendationEngine,
        model_version: str = "v1",
        device: str = "cuda",
    ) -> None:
        self.classifier = classifier
        self.ood_detector = ood_detector
        self.uncertainty_estimator = uncertainty_estimator
        self.calibrator = calibrator
        self.quality_assessor = quality_assessor
        self.explainer = explainer
        self.recommendation_engine = recommendation_engine
        self.model_version = model_version
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def predict(
        self,
        image_array: np.ndarray,           # Original raw image (H, W, 3) BGR
        input_tensor: torch.Tensor,        # Preprocessed tensor (1, C, H, W)
        generate_explanation: bool = True,
    ) -> PredictionResult:
        """
        Run the full trustworthy inference pipeline.

        Raises:
            ImageQualityError: if image quality is below threshold
            OODDetectedError: if OOD is detected (caller can catch and continue)
        """
        input_tensor = input_tensor.to(self.device)

        # Step 1: Image Quality Assessment
        quality = self.quality_assessor.assess(image_array)
        if not quality.is_acceptable:
            raise ImageQualityError(
                message="Image quality insufficient for reliable analysis",
                detail=str(quality.issues),
            )

        # Step 2: OOD Detection
        ood_result = self.ood_detector.detect(input_tensor)

        # Step 3: Classification
        cls_output = self.classifier.predict(input_tensor, self.CLASS_LABELS)

        # Step 4: Uncertainty Estimation
        uncertainty = self.uncertainty_estimator.estimate(input_tensor)

        # Step 5: Temperature-scaled confidence
        with torch.no_grad():
            calibrated_logits = self.calibrator(cls_output.logits)
            calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
            calibrated_confidence = float(calibrated_probs.max().item())

        # Step 6: Grad-CAM Explanation
        gradcam_output: GradCAMOutput | None = None
        if generate_explanation:
            original_float = image_array.astype(np.float32) / 255.0
            gradcam_output = self.explainer.explain(
                input_tensor=input_tensor,
                original_image=original_float,
                target_class=cls_output.predicted_class,
            )

        # Step 7: Clinical Recommendation
        predicted_enum = LesionClass(self.CLASS_LABELS[cls_output.predicted_class])
        recommendation = self.recommendation_engine.generate(
            predicted_class=predicted_enum,
            confidence=calibrated_confidence,
            uncertainty=uncertainty.predictive_entropy,
            is_ood=(ood_result.decision == "OUT_OF_DISTRIBUTION"),
            model_version=self.model_version,
        )

        # Build probabilities dict
        probs_np = cls_output.probabilities.cpu().numpy().squeeze()
        prob_dict = {
            label: float(prob)
            for label, prob in zip(self.CLASS_LABELS, probs_np, strict=False)
        }

        return PredictionResult(
            predicted_class=self.CLASS_LABELS[cls_output.predicted_class],
            predicted_class_index=cls_output.predicted_class,
            probabilities=prob_dict,
            raw_confidence=cls_output.confidence,
            calibrated_confidence=calibrated_confidence,
            quality_report=quality,
            ood_result=ood_result,
            uncertainty=uncertainty,
            gradcam=gradcam_output,
            recommendation=recommendation,
            model_version=self.model_version,
            inference_backend="pytorch",
        )
