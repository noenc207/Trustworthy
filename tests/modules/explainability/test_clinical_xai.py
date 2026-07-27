import numpy as np
import pytest
import torch
import torch.nn as nn

from src.infrastructure.ml_backends.torch.adapter import TorchBackendAdapter
from src.modules.classifier.result import PredictionResult
from src.modules.explainability.config import ExplainabilityConfig
from src.modules.explainability.enums import XAIAlgorithm
from src.modules.explainability.exceptions import ExplanationValidationError
from src.modules.explainability.orchestrator import ExplainabilityEngine


class MockCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, 2)
    def forward(self, x):
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)

def test_engine_initialization():
    adapter = TorchBackendAdapter()
    config = ExplainabilityConfig(primary_algorithm=XAIAlgorithm.GRADCAM)
    engine = ExplainabilityEngine(config, adapter)
    assert engine is not None

def test_validator_guards():
    from src.modules.explainability.metrics.validator import ExplanationValidator
    with pytest.raises(ExplanationValidationError, match="entirely zero"):
        ExplanationValidator.validate_heatmap(np.zeros((10, 10)))

    with pytest.raises(ExplanationValidationError, match="variance is near zero"):
        ExplanationValidator.validate_heatmap(np.ones((10, 10)))

def test_full_explainability_pipeline():
    adapter = TorchBackendAdapter()
    # Disable sanity for fast test
    config = ExplainabilityConfig(
        primary_algorithm=XAIAlgorithm.GRADCAM,
        enable_consensus=False,
        enable_faithfulness=True,
        enable_stability=False,
        enable_sanity_checks=False
    )
    engine = ExplainabilityEngine(config, adapter)
    model = MockCNN()

    # Initialize so we don't get 0-variance
    with torch.no_grad():
        model.layer4[0].weight.fill_(0.1)
        model.layer4[0].bias.fill_(0.0)
        model.classifier.weight.fill_(0.1)
        model.classifier.bias.fill_(0.0)

    image = np.ones((64, 64, 3), dtype=np.uint8) * 100
    image[20:40, 20:40] = 200 # lesion

    tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    pred = PredictionResult("MEL", 0, 0.9, {"MEL":0.9}, [])

    result = engine.evaluate(model, image, tensor, pred)

    assert result.is_valid is True
    assert result.heatmap is not None
    assert result.medical_metrics is not None
    assert result.faithfulness is not None
