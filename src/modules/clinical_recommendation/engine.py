"""
Clinical Recommendation Engine.

Translates AI classification and uncertainty outputs into
actionable clinical recommendations for healthcare providers.

IMPORTANT:
  This module generates DECISION SUPPORT — NOT clinical diagnosis.
  All recommendations must be reviewed by a qualified dermatologist.
  The AI system is an assistive tool only.

Recommendation logic:
  - Based on predicted class, confidence, urgency level
  - Considers uncertainty — high uncertainty → escalate to human expert
  - Considers OOD score — OOD images cannot be reliably diagnosed
  - Generates patient-friendly and clinician-facing reports
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.core.constants import LesionClass, UrgencyLevel, LESION_CLASS_NAMES, LESION_URGENCY


@dataclass
class ClinicalRecommendation:
    """Structured clinical recommendation."""
    # Decision support summary
    predicted_diagnosis: str
    urgency_level: str
    confidence_level: str           # 'high', 'moderate', 'low'
    recommendation_text: str        # Clinician-facing recommendation
    patient_summary: str            # Patient-friendly explanation
    next_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_version: str = "unknown"
    disclaimer: str = (
        "This AI-generated recommendation is a decision support tool only. "
        "It does not constitute a medical diagnosis. "
        "Always consult a qualified dermatologist."
    )


class ClinicalRecommendationEngine:
    """
    Generates clinical recommendations from AI analysis outputs.

    Inputs:
      - predicted_class: LesionClass enum
      - confidence: float [0, 1]
      - uncertainty: float (predictive entropy)
      - is_ood: bool

    Output:
      - ClinicalRecommendation with structured advice
    """

    HIGH_CONFIDENCE_THRESHOLD = 0.80
    MODERATE_CONFIDENCE_THRESHOLD = 0.60
    HIGH_UNCERTAINTY_THRESHOLD = 0.5

    def generate(
        self,
        predicted_class: LesionClass,
        confidence: float,
        uncertainty: float,
        is_ood: bool,
        model_version: str = "unknown",
    ) -> ClinicalRecommendation:
        """Generate a structured clinical recommendation."""
        warnings: list[str] = []
        next_steps: list[str] = []

        # OOD flag overrides all normal logic
        if is_ood:
            return ClinicalRecommendation(
                predicted_diagnosis="Undetermined — Out-of-Distribution Input",
                urgency_level="unknown",
                confidence_level="low",
                recommendation_text=(
                    "The submitted image is outside the training distribution of the AI model. "
                    "A reliable prediction cannot be made. "
                    "Please submit a higher quality dermoscopic image or refer directly to a dermatologist."
                ),
                patient_summary="The AI could not analyze this image. Please see a doctor.",
                next_steps=["Retake image with better lighting/focus", "Refer to dermatologist"],
                warnings=["Out-of-distribution input detected"],
                model_version=model_version,
            )

        # Determine confidence level label
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            confidence_label = "high"
        elif confidence >= self.MODERATE_CONFIDENCE_THRESHOLD:
            confidence_label = "moderate"
        else:
            confidence_label = "low"

        # High uncertainty flag
        if uncertainty > self.HIGH_UNCERTAINTY_THRESHOLD:
            warnings.append(
                f"High model uncertainty detected (entropy={uncertainty:.3f}). "
                "Human expert review is strongly recommended."
            )
            next_steps.append("Refer to dermatologist for expert review")

        diagnosis_name = LESION_CLASS_NAMES[predicted_class]
        urgency = LESION_URGENCY[predicted_class]

        recommendation = self._get_recommendation(predicted_class, confidence_label, urgency)
        patient_summary = self._get_patient_summary(predicted_class, confidence_label)
        next_steps.extend(self._get_next_steps(predicted_class, urgency))

        return ClinicalRecommendation(
            predicted_diagnosis=diagnosis_name,
            urgency_level=urgency.value,
            confidence_level=confidence_label,
            recommendation_text=recommendation,
            patient_summary=patient_summary,
            next_steps=list(dict.fromkeys(next_steps)),  # Deduplicate
            warnings=warnings,
            model_version=model_version,
        )

    def _get_recommendation(self, cls: LesionClass, confidence: str, urgency: object) -> str:
        urgency_map = {
            "critical": "IMMEDIATE referral to dermatologist recommended.",
            "high": "Referral to dermatologist recommended within 2 weeks.",
            "moderate": "Monitor lesion and schedule follow-up appointment.",
            "low": "Routine monitoring. No immediate action required.",
        }
        urgency_str = str(urgency.value) if hasattr(urgency, "value") else str(urgency)
        return (
            f"AI analysis suggests {LESION_CLASS_NAMES[cls]} "
            f"with {confidence} confidence. {urgency_map.get(urgency_str, '')}"
        )

    def _get_patient_summary(self, cls: LesionClass, confidence: str) -> str:
        friendly_map = {
            LesionClass.MEL: "The AI has detected features consistent with melanoma, a serious skin condition.",
            LesionClass.NV: "The AI has detected features of a common mole.",
            LesionClass.BCC: "The AI has detected features consistent with basal cell carcinoma.",
            LesionClass.AKIEC: "The AI has detected features of a pre-cancerous skin condition.",
            LesionClass.BKL: "The AI has detected features of a benign skin growth.",
            LesionClass.DF: "The AI has detected features of a benign fibrous skin growth.",
            LesionClass.VASC: "The AI has detected features of a vascular skin lesion.",
        }
        return friendly_map.get(cls, "AI analysis completed.")

    def _get_next_steps(self, cls: LesionClass, urgency: object) -> list[str]:
        urgency_str = str(urgency.value) if hasattr(urgency, "value") else str(urgency)
        if urgency_str == "critical":
            return [
                "Schedule urgent dermatology appointment",
                "Do not delay seeking medical attention",
                "Photograph lesion for monitoring changes",
            ]
        elif urgency_str == "high":
            return [
                "Schedule dermatology appointment within 2 weeks",
                "Monitor lesion for changes in size, color, or shape",
            ]
        else:
            return [
                "Monitor lesion regularly using the ABCDE rule",
                "Re-assess at annual skin check",
            ]
