"""
Unit tests for OOD Detection module.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


@pytest.fixture()
def simple_model():
    """Minimal model for testing."""
    return nn.Linear(3 * 224 * 224, 7)


@pytest.mark.unit
class TestOODDetector:
    """Unit tests for OODDetector."""

    def test_msp_detect_returns_result(self, simple_model):
        """MSP detection should return an OODResult."""
        from src.modules.ood_detection.detector import OODDetector, OODMethod, OODResult

        class FlatModel(nn.Module):
            def __init__(self, inner): super().__init__(); self.inner = inner
            def forward(self, x): return self.inner(x.view(x.size(0), -1))

        model = FlatModel(simple_model)
        detector = OODDetector(model, method=OODMethod.MSP, threshold=0.5)
        x = torch.randn(1, 3, 224, 224)
        result = detector.detect(x)
        assert isinstance(result, OODResult)
        assert isinstance(result.is_ood, bool)
        assert 0.0 <= result.ood_score <= 1.0

    def test_ood_score_range(self, simple_model):
        """OOD score must be in [0, 1]."""
        from src.modules.ood_detection.detector import OODDetector, OODMethod

        class FlatModel(nn.Module):
            def __init__(self, inner): super().__init__(); self.inner = inner
            def forward(self, x): return self.inner(x.view(x.size(0), -1))

        model = FlatModel(simple_model)
        detector = OODDetector(model, method=OODMethod.MSP)
        x = torch.randn(1, 3, 224, 224)
        result = detector.detect(x)
        assert 0.0 <= result.ood_score <= 1.0
