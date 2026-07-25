"""Framework abstract interfaces."""
from __future__ import annotations

from .base_calibrator import AbstractCalibrator
from .base_classifier import AbstractClassifier
from .base_explainer import AbstractExplainer
from .base_model import AbstractModel
from .base_ood import AbstractOODDetector
from .base_quality import AbstractQualityAssessor
from .base_recommender import AbstractRecommendationEngine
from .base_segmentor import AbstractSegmentor
from .base_uncertainty import AbstractUncertaintyEstimator

__all__ = [
    "AbstractCalibrator",
    "AbstractClassifier",
    "AbstractExplainer",
    "AbstractModel",
    "AbstractOODDetector",
    "AbstractQualityAssessor",
    "AbstractRecommendationEngine",
    "AbstractSegmentor",
    "AbstractUncertaintyEstimator",
]
