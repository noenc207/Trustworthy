"""
TrustDerm AI Core Framework.
Provides base interfaces, metrics, and common types for AI pipelines.
"""
from __future__ import annotations

from .common import PipelineResult
from .interfaces import (
    AbstractCalibrator,
    AbstractClassifier,
    AbstractExplainer,
    AbstractModel,
    AbstractOODDetector,
    AbstractQualityAssessor,
    AbstractRecommendationEngine,
    AbstractSegmentor,
    AbstractUncertaintyEstimator,
)
from .metrics import ExecutionMetadata, ResourceMetrics, TimingMetrics

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
    "ExecutionMetadata",
    "PipelineResult",
    "ResourceMetrics",
    "TimingMetrics",
]
