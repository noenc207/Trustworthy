"""
Project-wide constants: class labels, clinical mappings, thresholds.
"""
from __future__ import annotations

from enum import StrEnum


class LesionClass(StrEnum):
    """HAM10000 / ISIC lesion class labels."""
    MEL   = "mel"      # Melanoma
    NV    = "nv"       # Melanocytic nevi
    BCC   = "bcc"      # Basal cell carcinoma
    AKIEC = "akiec"    # Actinic keratosis / Bowen's disease
    BKL   = "bkl"      # Benign keratosis
    DF    = "df"       # Dermatofibroma
    VASC  = "vasc"     # Vascular lesions


# Maps class enum to full clinical name
LESION_CLASS_NAMES: dict[LesionClass, str] = {
    LesionClass.MEL:   "Melanoma",
    LesionClass.NV:    "Melanocytic Nevi",
    LesionClass.BCC:   "Basal Cell Carcinoma",
    LesionClass.AKIEC: "Actinic Keratosis / Intraepithelial Carcinoma",
    LesionClass.BKL:   "Benign Keratosis-like Lesions",
    LesionClass.DF:    "Dermatofibroma",
    LesionClass.VASC:  "Vascular Lesions",
}

# Clinical urgency levels
class UrgencyLevel(StrEnum):
    CRITICAL  = "critical"   # Immediate dermatologist referral
    HIGH      = "high"       # Referral within 2 weeks
    MODERATE  = "moderate"   # Monitor and follow up
    LOW       = "low"        # Routine monitoring

LESION_URGENCY: dict[LesionClass, UrgencyLevel] = {
    LesionClass.MEL:   UrgencyLevel.CRITICAL,
    LesionClass.BCC:   UrgencyLevel.HIGH,
    LesionClass.AKIEC: UrgencyLevel.HIGH,
    LesionClass.NV:    UrgencyLevel.LOW,
    LesionClass.BKL:   UrgencyLevel.LOW,
    LesionClass.DF:    UrgencyLevel.LOW,
    LesionClass.VASC:  UrgencyLevel.MODERATE,
}

# Model architecture choices
SUPPORTED_ARCHITECTURES: list[str] = [
    "efficientnet_b0", "efficientnet_b4",
    "resnet50", "resnet101",
    "densenet121", "densenet201",
    "vit_base_patch16_224",
    "swin_base_patch4_window7_224",
]

# Image preprocessing constants
IMAGE_MEAN: tuple[float, float, float] = (0.763, 0.546, 0.570)
IMAGE_STD:  tuple[float, float, float] = (0.141, 0.152, 0.169)
DEFAULT_IMAGE_SIZE: int = 224

# Minimum quality score for valid images
MIN_QUALITY_SCORE: float = 0.5
