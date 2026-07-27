"""
Image Quality Assessment Module.

Evaluates dermoscopic image quality before feeding to AI models.
Low-quality images produce unreliable predictions.

Checks:
  - Blur detection (Laplacian variance)
  - Exposure (brightness histogram analysis)
  - Color balance (RGB channel distribution)
  - Artifact detection (ruler marks, hair artifacts)
  - Resolution validation
  - Overall composite quality score
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityReport:
    """Comprehensive image quality assessment report."""
    overall_score: float               # Composite score in [0, 1]
    is_acceptable: bool                # Whether quality meets minimum threshold
    blur_score: float                  # 0=very blurry, 1=sharp
    exposure_score: float              # 0=under/over-exposed, 1=well-exposed
    color_score: float                 # 0=poor color, 1=good color balance
    resolution_score: float            # 0=too small, 1=adequate resolution
    artifact_score: float              # 0=many artifacts, 1=clean
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class ImageQualityAssessor:
    """
    Automated dermoscopic image quality assessor.

    Args:
        min_quality_score: Minimum acceptable composite score
        min_resolution: Minimum width/height in pixels
        blur_threshold: Laplacian variance threshold (lower = more blurry)
    """

    def __init__(
        self,
        min_quality_score: float = 0.5,
        min_resolution: int = 224,
        blur_threshold: float = 100.0,
    ) -> None:
        self.min_quality_score = min_quality_score
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold

    def assess(self, image: np.ndarray) -> QualityReport:
        """
        Run full quality assessment pipeline on an image.

        Args:
            image: Input image as numpy array (H, W, 3) BGR or RGB

        Returns:
            QualityReport with scores and recommendations
        """
        issues: list[str] = []
        recommendations: list[str] = []

        blur_score = self._check_blur(image)
        exposure_score = self._check_exposure(image)
        color_score = self._check_color_balance(image)
        resolution_score = self._check_resolution(image)
        artifact_score = self._check_artifacts(image)

        if blur_score < 0.3:
            issues.append("Image appears blurry")
            recommendations.append("Capture image with better focus")
        if exposure_score < 0.3:
            issues.append("Poor exposure (too bright or too dark)")
            recommendations.append("Adjust lighting conditions")
        if resolution_score < 0.5:
            issues.append(f"Resolution too low (minimum {self.min_resolution}px)")
            recommendations.append("Use higher resolution camera")

        # Weighted composite score
        overall = (
            0.30 * blur_score
            + 0.25 * exposure_score
            + 0.20 * color_score
            + 0.15 * resolution_score
            + 0.10 * artifact_score
        )

        return QualityReport(
            overall_score=float(overall),
            is_acceptable=overall >= self.min_quality_score,
            blur_score=float(blur_score),
            exposure_score=float(exposure_score),
            color_score=float(color_score),
            resolution_score=float(resolution_score),
            artifact_score=float(artifact_score),
            issues=issues,
            recommendations=recommendations,
        )

    def _check_blur(self, image: np.ndarray) -> float:
        """Laplacian variance — higher variance = sharper image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(variance / self.blur_threshold, 1.0))

    def _check_exposure(self, image: np.ndarray) -> float:
        """Histogram-based exposure check."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mean_brightness = float(gray.mean())
        # Acceptable range: 60-200 (on 0-255 scale)
        if 60 <= mean_brightness <= 200:
            # Score based on distance from ideal 128
            return 1.0 - abs(mean_brightness - 128) / 128
        return 0.0

    def _check_color_balance(self, image: np.ndarray) -> float:
        """Check RGB channel balance."""
        if image.ndim != 3:
            return 1.0
        channel_means = image.mean(axis=(0, 1))
        cv = np.std(channel_means) / (np.mean(channel_means) + 1e-8)
        return float(max(0.0, 1.0 - cv))

    def _check_resolution(self, image: np.ndarray) -> float:
        """Check if image meets minimum resolution."""
        h, w = image.shape[:2]
        min_dim = min(h, w)
        if min_dim >= self.min_resolution:
            return 1.0
        return float(min_dim / self.min_resolution)

    def _check_artifacts(self, image: np.ndarray) -> float:
        """
        Evaluate the image for artifacts like hair, marker ink, or surgical markings.
        Returns a score from 0.0 (many artifacts) to 1.0 (clean).
        """
        # Simple heuristic: check for very dark or bright regions
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        dark_ratio = (gray < 10).mean()
        bright_ratio = (gray > 250).mean()
        artifact_ratio = dark_ratio + bright_ratio
        return float(max(0.0, 1.0 - artifact_ratio * 10))
