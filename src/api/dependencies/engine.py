"""
Inference Engine Dependency — Phase 5 upgraded architecture.

Wires all concrete AI module implementations into the new
PipelineExecutor-based architecture using the Stage/Protocol pattern.

The singleton PipelineExecutor is provided to PredictionService via FastAPI DI.
The legacy TrustworthyInferenceEngine is preserved for backwards compatibility
during the transition — PredictionService will be updated to use the new executor.
"""
from __future__ import annotations

from fastapi import HTTPException, status
from loguru import logger

from src.modules.calibration.calibrator import TemperatureScaling
from src.modules.classification.classifier import SkinLesionClassifier
from src.modules.clinical_recommendation.engine import ClinicalRecommendationEngine

# Legacy import kept for PredictionService backwards compatibility
from src.modules.inference_engine.engine import TrustworthyInferenceEngine
from src.modules.inference_engine.executor import PipelineExecutor
from src.modules.inference_engine.module_registry import AIModuleRegistry
from src.modules.inference_engine.stages import (
    CalibrationStage,
    ClassificationStage,
    ClinicalRecommendationStage,
    OODDetectionStage,
    PreprocessingStage,
    QualityAssessmentStage,
    UncertaintyEstimationStage,
)
from src.modules.ood_detection.detector import OODDetector, OODMethod
from src.modules.quality_assessment.assessor import ImageQualityAssessor
from src.modules.uncertainty.estimator import MCDropoutEstimator

_executor: PipelineExecutor | None = None
_legacy_engine: TrustworthyInferenceEngine | None = None
_module_registry: AIModuleRegistry | None = None


def _build_preprocessor(image_size: int = 224):
    """
    Build a minimal fallback preprocessor (no Albumentations required).
    In production, replace this with PreprocessingFactory.build(config.preprocessing.inference).
    """
    import cv2
    import numpy as np
    import torch

    from src.core.constants import IMAGE_MEAN, IMAGE_STD

    def preprocess(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        resized = cv2.resize(rgb, (image_size, image_size))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - np.array(IMAGE_MEAN)) / np.array(IMAGE_STD)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
        return tensor

    return preprocess


def _build_executor() -> PipelineExecutor:
    """
    Instantiate all concrete modules, register them, and compose the pipeline.
    Uses untrained/dummy weights on CPU for safe integration without real model files.
    """
    # 1. Instantiate concrete modules
    classifier = SkinLesionClassifier(pretrained=False)
    ood_detector = OODDetector(classifier, method=OODMethod.ENERGY, threshold=0.5)
    uncertainty_estimator = MCDropoutEstimator(classifier, num_samples=10)
    calibrator = TemperatureScaling()
    quality_assessor = ImageQualityAssessor()
    recommendation_engine = ClinicalRecommendationEngine()
    preprocessor = _build_preprocessor()

    # 2. Register in the global AI Module Registry
    registry = AIModuleRegistry.instance()
    registry.register_classifier("efficientnet_b4", classifier)
    registry.register_ood_detector("energy", ood_detector)
    registry.register_uncertainty_estimator("mc_dropout", uncertainty_estimator)
    registry.register_calibrator("temperature_scaling", calibrator)
    registry.register_quality_assessor("heuristic", quality_assessor)
    registry.register_recommendation_engine("rule_based", recommendation_engine)
    registry.register_preprocessor("default", preprocessor)

    logger.info(f"AIModuleRegistry populated: {registry.summary()}")

    # 3. Compose the ordered pipeline stages
    stages = [
        QualityAssessmentStage(assessor=quality_assessor),
        PreprocessingStage(preprocessor=preprocessor),
        ClassificationStage(classifier=classifier),
        OODDetectionStage(detector=ood_detector),
        UncertaintyEstimationStage(estimator=uncertainty_estimator),
        CalibrationStage(calibrator=calibrator),
        # ExplainabilityStage requires a compatible target_layer — skip for integration stub
        # ExplainabilityStage(explainer=explainer),
        ClinicalRecommendationStage(engine=recommendation_engine),
    ]

    return PipelineExecutor(stages=stages)


async def get_pipeline_executor() -> PipelineExecutor:
    """FastAPI dependency: returns the singleton PipelineExecutor."""
    global _executor
    if _executor is None:
        try:
            _executor = _build_executor()
            logger.info("PipelineExecutor initialized successfully")
        except Exception as exc:
            logger.exception("Failed to initialize PipelineExecutor")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline executor initialization failed: {exc}",
            ) from exc
    return _executor


async def get_inference_engine() -> TrustworthyInferenceEngine:
    """
    FastAPI dependency: returns the legacy TrustworthyInferenceEngine.
    Kept for backward compatibility with PredictionService.
    Will be removed once PredictionService migrates to PipelineExecutor.
    """
    global _legacy_engine
    if _legacy_engine is None:
        try:
            classifier = SkinLesionClassifier(pretrained=False)
            ood = OODDetector(classifier)
            uncert = MCDropoutEstimator(classifier)
            calib = TemperatureScaling()
            qual = ImageQualityAssessor()
            rec = ClinicalRecommendationEngine()

            _legacy_engine = TrustworthyInferenceEngine(
                classifier=classifier,
                ood_detector=ood,
                uncertainty_estimator=uncert,
                calibrator=calib,
                quality_assessor=qual,
                explainer=None,
                recommendation_engine=rec,
                device="cpu",
            )
            logger.info("Legacy TrustworthyInferenceEngine initialized (compatibility mode)")
        except Exception as exc:
            logger.exception("Failed to initialize legacy inference engine")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Engine initialization failed: {exc}",
            ) from exc
    return _legacy_engine


def get_module_registry() -> AIModuleRegistry:
    """FastAPI dependency: returns the global AI module registry."""
    return AIModuleRegistry.instance()
