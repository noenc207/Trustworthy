"""
Unit tests for the SkinLesionClassifier module.
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
class TestSkinLesionClassifier:
    """Unit tests for SkinLesionClassifier."""

    @pytest.fixture()
    def classifier(self):
        """Create a small test classifier."""
        from src.modules.classification.classifier import SkinLesionClassifier
        return SkinLesionClassifier(
            backbone="efficientnet_b0",
            num_classes=7,
            pretrained=False,
        )

    def test_forward_shape(self, classifier):
        """Output shape should be (batch_size, num_classes)."""
        x = torch.randn(4, 3, 224, 224)
        output = classifier(x)
        assert output.shape == (4, 7)

    def test_predict_returns_dataclass(self, classifier):
        """predict() should return ClassificationOutput."""
        from src.modules.classification.classifier import ClassificationOutput
        x = torch.randn(1, 3, 224, 224)
        labels = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
        result = classifier.predict(x, labels)
        assert isinstance(result, ClassificationOutput)
        assert 0 <= result.predicted_class < 7
        assert 0.0 <= result.confidence <= 1.0

    def test_probabilities_sum_to_one(self, classifier):
        """Softmax probabilities must sum to 1 (within floating point tolerance)."""
        x = torch.randn(1, 3, 224, 224)
        labels = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
        result = classifier.predict(x, labels)
        prob_sum = result.probabilities.sum().item()
        assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities sum to {prob_sum}, expected ~1.0"

    def test_eval_mode_during_predict(self, classifier):
        """Classifier must be in eval mode during predict."""
        x = torch.randn(1, 3, 224, 224)
        labels = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
        classifier.predict(x, labels)
        assert not classifier.training
