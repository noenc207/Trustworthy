"""
Unit tests for Confidence Calibration module.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.unit
class TestTemperatureScaling:
    """Unit tests for TemperatureScaling calibrator."""

    @pytest.fixture()
    def calibrator(self):
        from src.modules.calibration.calibrator import TemperatureScaling
        return TemperatureScaling()

    def test_initial_temperature_is_one(self, calibrator):
        """Temperature should initialize to 1.0 (no scaling)."""
        assert abs(calibrator.temperature.item() - 1.0) < 1e-6

    def test_forward_scales_logits(self, calibrator):
        """At T=2, logits should be halved."""
        calibrator.temperature.data = torch.tensor([2.0])
        logits = torch.tensor([[2.0, 4.0, 6.0]])
        scaled = calibrator(logits)
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(scaled, expected, atol=1e-5)

    def test_ece_perfect_calibration(self, calibrator):
        """Perfect calibration should yield ECE near 0."""
        # Perfect case: 100% confidence = 100% accuracy
        probs = np.eye(7)  # One-hot probabilities
        labels = np.arange(7)
        ece = calibrator.compute_ece(probs, labels)
        assert ece < 0.01, f"ECE should be near 0, got {ece}"

    def test_ece_range(self, calibrator):
        """ECE should be in [0, 1]."""
        probs = np.random.dirichlet(np.ones(7), size=100)
        labels = np.random.randint(0, 7, size=100)
        ece = calibrator.compute_ece(probs, labels)
        assert 0.0 <= ece <= 1.0
