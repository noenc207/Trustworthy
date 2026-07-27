"""
Unit tests for Image Quality Assessment.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
class TestImageQualityAssessor:
    """Unit tests for ImageQualityAssessor."""

    @pytest.fixture()
    def assessor(self):
        from src.modules.quality_assessment.assessor import ImageQualityAssessor
        return ImageQualityAssessor(min_quality_score=0.5, min_resolution=64)

    def test_sharp_image_passes(self, assessor):
        """A high-resolution sharp image should pass quality check."""
        # Create a sharp synthetic image (high frequency content)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        report = assessor.assess(image)
        assert isinstance(report.overall_score, float)
        assert 0.0 <= report.overall_score <= 1.0

    def test_tiny_image_fails_resolution(self, assessor):
        """Image below minimum resolution should have low resolution score."""
        small_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        report = assessor.assess(small_image)
        assert report.resolution_score < 1.0

    def test_report_structure(self, assessor):
        """Report should contain all expected fields."""
        from src.modules.quality_assessment.assessor import QualityReport
        image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        report = assessor.assess(image)
        assert isinstance(report, QualityReport)
        assert hasattr(report, "overall_score")
        assert hasattr(report, "blur_score")
        assert hasattr(report, "exposure_score")
        assert isinstance(report.issues, list)
