"""
Unit tests for Uncertainty Estimation module.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


@pytest.mark.unit
class TestMCDropoutEstimator:
    """Unit tests for MCDropoutEstimator."""

    @pytest.fixture()
    def model_with_dropout(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def test_estimate_returns_output(self, model_with_dropout):
        """estimate() should return UncertaintyOutput."""
        from src.modules.uncertainty.estimator import MCDropoutEstimator, UncertaintyOutput
        estimator = MCDropoutEstimator(model_with_dropout, num_samples=5)
        x = torch.randn(1, 3, 32, 32)
        result = estimator.estimate(x)
        assert isinstance(result, UncertaintyOutput)

    def test_probabilities_shape(self, model_with_dropout):
        """Mean probabilities should have shape (num_classes,)."""
        from src.modules.uncertainty.estimator import MCDropoutEstimator
        estimator = MCDropoutEstimator(model_with_dropout, num_samples=5)
        x = torch.randn(1, 3, 32, 32)
        result = estimator.estimate(x)
        assert result.mean_probabilities.shape == (7,)

    def test_entropy_non_negative(self, model_with_dropout):
        """Predictive entropy must be non-negative."""
        from src.modules.uncertainty.estimator import MCDropoutEstimator
        estimator = MCDropoutEstimator(model_with_dropout, num_samples=5)
        x = torch.randn(1, 3, 32, 32)
        result = estimator.estimate(x)
        assert result.predictive_entropy >= 0.0

    def test_epistemic_non_negative(self, model_with_dropout):
        """Epistemic uncertainty must be non-negative."""
        from src.modules.uncertainty.estimator import MCDropoutEstimator
        estimator = MCDropoutEstimator(model_with_dropout, num_samples=5)
        x = torch.randn(1, 3, 32, 32)
        result = estimator.estimate(x)
        assert result.epistemic_uncertainty >= 0.0
