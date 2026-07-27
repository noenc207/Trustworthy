"""
Pydantic schemas for prediction API.

Separating schemas from DB models follows the Repository pattern.
Schemas define the API contract (what gets serialized to/from JSON).
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    image_id: UUID = Field(..., description="UUID of the previously uploaded image")
    generate_explanation: bool = Field(
        default=True, description="Whether to generate Grad-CAM explanation"
    )
    ood_method: str = Field(
        default="energy", description="OOD detection method: 'msp', 'energy', 'mahalanobis'"
    )
    uncertainty_samples: int = Field(
        default=30, ge=10, le=100,
        description="Number of MC Dropout forward passes for uncertainty estimation"
    )


class ClassProbability(BaseModel):
    """Single class probability entry."""
    class_name: str
    probability: float = Field(..., ge=0.0, le=1.0)


class UncertaintyEstimate(BaseModel):
    """Uncertainty decomposition in prediction response."""
    predictive_entropy: float
    mutual_information: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    num_samples: int


class OODResultSchema(BaseModel):
    """OOD detection result in prediction response."""
    is_ood: bool
    ood_score: float
    method: str
    confidence: float


class QualityReportSchema(BaseModel):
    """Image quality report in prediction response."""
    overall_score: float
    is_acceptable: bool
    blur_score: float
    exposure_score: float
    resolution_score: float
    issues: list[str]


class RecommendationSchema(BaseModel):
    """Clinical recommendation in prediction response."""
    predicted_diagnosis: str
    urgency_level: str
    confidence_level: str
    recommendation_text: str
    patient_summary: str
    next_steps: list[str]
    warnings: list[str]
    disclaimer: str


class PredictionResponse(BaseModel):
    """Full prediction response from the inference engine."""
    model_config = ConfigDict(from_attributes=True)

    # Identifiers
    prediction_id: UUID
    image_id: UUID
    # Core prediction
    predicted_class: str
    predicted_class_index: int
    probabilities: list[ClassProbability]
    raw_confidence: float
    calibrated_confidence: float
    # Trustworthiness signals
    quality: QualityReportSchema
    ood: OODResultSchema
    uncertainty: UncertaintyEstimate
    # Explainability
    gradcam_url: str | None = None      # URL to retrieve CAM visualization
    # Clinical
    recommendation: RecommendationSchema
    # Metadata
    model_version: str
    inference_backend: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)
