"""
Data Transfer Objects (DTOs) and common types for AI modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Backend-agnostic tensor type alias
TensorLike = Any


@dataclass
class QualityReport:
    """Image quality assessment result."""
    overall_score: float
    is_acceptable: bool
    blur_score: float = 0.0
    exposure_score: float = 0.0
    color_score: float = 0.0
    resolution_score: float = 0.0
    artifact_score: float = 0.0
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Classification output from any classifier."""
    logits: TensorLike
    probabilities: TensorLike
    predicted_class: int
    confidence: float
    class_labels: list[str]


@dataclass
class OODResult:
    """Out-of-Distribution detection result."""
    method: str
    score: float | None
    confidence: float | None
    uncertainty: float | None
    threshold: float | None
    decision: str
    valid: bool
    status: str
    latency_ms: float
    memory_mb: float
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UncertaintyResult:
    """Uncertainty estimation result."""
    mean_probabilities: Any
    predictive_entropy: float
    mutual_information: float
    aleatoric_variance: float | Any
    epistemic_variance: float | Any
    num_samples: int


@dataclass
class ExplanationResult:
    """Explainability output (heatmap / saliency map)."""
    heatmap: Any
    visualization: Any | None
    target_class: int
    method_name: str


@dataclass
class ClinicalRecommendation:
    """Structured clinical recommendation."""
    predicted_diagnosis: str
    urgency_level: str
    confidence_level: str
    recommendation_text: str
    patient_summary: str
    next_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_version: str = "unknown"
    disclaimer: str = (
        "This AI-generated recommendation is a decision support tool only. "
        "It does not constitute a medical diagnosis. "
        "Always consult a qualified dermatologist."
    )
