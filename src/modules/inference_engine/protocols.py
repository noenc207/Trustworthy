"""
AI Module Protocol Interfaces.

Defines structural typing contracts (PEP 544) for every AI module.
Any class that matches a protocol's method signatures is automatically
compatible — no inheritance required.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.modules.inference_engine.types import (
    ClassificationResult,
    ClinicalRecommendation,
    ExplanationResult,
    OODResult,
    QualityReport,
    TensorLike,
    UncertaintyResult,
)


@runtime_checkable
class ImageQualityAssessorProtocol(Protocol):
    """Contract for image quality assessment modules."""

    def assess(self, image: Any) -> QualityReport:
        """Assess the quality of a raw image."""
        ...


@runtime_checkable
class PreprocessorProtocol(Protocol):
    """Contract for preprocessing pipeline modules."""

    def __call__(self, image: Any) -> TensorLike:
        """Transform a raw image into a normalized model-input tensor."""
        ...


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Contract for classification modules."""

    def predict(self, tensor: TensorLike) -> ClassificationResult:
        """Run classification on a preprocessed tensor."""
        ...

    def forward(self, x: TensorLike) -> TensorLike:
        """Raw forward pass returning logits."""
        ...


@runtime_checkable
class OODDetectorProtocol(Protocol):
    """Contract for OOD detection modules."""

    def detect(self, tensor: TensorLike) -> OODResult:
        """Run OOD detection on a preprocessed input tensor."""
        ...


@runtime_checkable
class UncertaintyEstimatorProtocol(Protocol):
    """Contract for uncertainty estimation modules."""

    def estimate(self, tensor: TensorLike) -> UncertaintyResult:
        """Estimate uncertainty via stochastic forward passes or ensembles."""
        ...


@runtime_checkable
class CalibratorProtocol(Protocol):
    """Contract for confidence calibration modules."""

    def forward(self, logits: TensorLike) -> TensorLike:
        """Return calibrated logits."""
        ...


@runtime_checkable
class ExplainerProtocol(Protocol):
    """Contract for explainability modules."""

    def explain(
        self,
        input_tensor: TensorLike,
        original_image: Any,
        target_class: int | None = None,
    ) -> ExplanationResult:
        """Generate a visual explanation for a prediction."""
        ...


@runtime_checkable
class RecommendationEngineProtocol(Protocol):
    """Contract for clinical recommendation engines."""

    def generate(
        self,
        predicted_class: Any,
        confidence: float,
        uncertainty: float,
        is_ood: bool,
        model_version: str = "unknown",
    ) -> ClinicalRecommendation:
        """Generate a structured clinical recommendation."""
        ...
