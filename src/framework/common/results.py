"""
Unified pipeline result DTO.
Replaces all previous PredictionResult variants.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from src.framework.metrics.execution import ExecutionMetadata

# Prevent circular imports while maintaining strict typing
if TYPE_CHECKING:
    from src.modules.inference_engine.types import (
        ClinicalRecommendation,
        ExplanationResult,
        OODResult,
        QualityReport,
        UncertaintyResult,
    )


@dataclass
class PipelineResult:
    """Unified response object encapsulating all inference pipeline outputs."""

    # Identity
    request_id: str
    model_version: str
    pipeline_version: str

    # Core prediction
    predicted_class: str
    predicted_class_index: int
    class_probabilities: dict[str, float]
    raw_confidence: float
    calibrated_confidence: float

    # Trustworthiness signals
    quality: QualityReport | None
    ood: OODResult | None
    uncertainty: UncertaintyResult | None

    # Explainability
    explanation: ExplanationResult | None

    # Clinical
    recommendation: ClinicalRecommendation | None

    # Execution metadata
    execution: ExecutionMetadata

    # Observability
    warnings: list[str]
    errors: list[str]

    def is_trustworthy(self, ood_threshold: float = 0.5, uncertainty_threshold: float = 0.5) -> bool:
        """Determine if the prediction can be fully trusted."""
        if self.quality and not self.quality.is_acceptable:
            return False
        if self.ood and self.ood.is_ood:
            return False
        return not (self.uncertainty and self.uncertainty.predictive_entropy > uncertainty_threshold)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary summary."""
        return asdict(self)

    def has_errors(self) -> bool:
        """Return True if any errors occurred during pipeline execution."""
        return len(self.errors) > 0
