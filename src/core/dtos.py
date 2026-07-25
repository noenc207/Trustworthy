"""Data Transfer Objects for the core application.

This module provides frozen dataclasses representing the core data structures
used to transfer data between different layers of the application.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionRequestDTO:
    """Data Transfer Object for a prediction request.

    Attributes:
        image_data: The raw image bytes representing the skin lesion.
        patient_id: An optional identifier for the patient.
        metadata: Optional dictionary containing additional information (e.g., age, sex).
    """

    image_data: bytes
    patient_id: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PredictionResponseDTO:
    """Data Transfer Object for a prediction response.

    Attributes:
        prediction_id: Unique identifier for this prediction.
        label: The predicted class label (e.g., 'melanoma', 'benign').
        confidence: The confidence score of the prediction (0.0 to 1.0).
        explanation_map: Optional bytes representing a saliency map or heatmap.
    """

    prediction_id: str
    label: str
    confidence: float
    explanation_map: bytes | None = None
