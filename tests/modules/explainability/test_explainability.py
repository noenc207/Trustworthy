import numpy as np
import pytest
import torch.nn as nn

from src.infrastructure.ml_backends.torch.adapter import TorchBackendAdapter
from src.modules.explainability.cache import ExplainabilityCache
from src.modules.explainability.config import ExplainabilityConfig
from src.modules.explainability.hook_manager import HookManager
from src.modules.explainability.layer_resolver import LayerResolver
from src.modules.explainability.metrics import ExplainabilityMetrics
from src.modules.explainability.result import ExplainabilityResult
from src.modules.explainability.validator import ExplanationValidationError, ExplanationValidator


class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def test_layer_resolver():
    model = DummyCNN()
    # It should fallback to finding "features"
    layer = LayerResolver.resolve(model)
    assert layer == "features"

def test_validator_nan_inf():
    with pytest.raises(ExplanationValidationError, match="NaN detected"):
        ExplanationValidator.validate_tensors(np.array([np.nan]), np.array([1.0]))

    with pytest.raises(ExplanationValidationError, match="Inf detected"):
        ExplanationValidator.validate_tensors(np.array([np.inf]), np.array([1.0]))

    with pytest.raises(ExplanationValidationError, match="variance is near zero"):
        ExplanationValidator.validate_tensors(np.array([1.0, 1.0]), np.array([1.0, 1.0]))

def test_metrics_empty():
    res = ExplainabilityMetrics.compute_all(np.array([]))
    assert res == {}

def test_metrics_computation():
    heatmap = np.zeros((10, 10))
    heatmap[2:5, 2:5] = 1.0
    res = ExplainabilityMetrics.compute_all(heatmap)
    assert res["peak_activation"] == 1.0
    assert res["activation_area_pct"] == 9.0  # 9 pixels out of 100
    assert res["center_of_mass"] == (3.0, 3.0)

def test_hook_manager_cleanup():
    model = DummyCNN()
    adapter = TorchBackendAdapter()
    hm = HookManager(adapter, model)
    hm.register_hooks("features")

    assert len(hm._hook_handles) == 2
    hm.cleanup()
    assert len(hm._hook_handles) == 0
    assert hm.activations is None

def test_cache():
    cache = ExplainabilityCache()
    config = ExplainabilityConfig()
    res = ExplainabilityResult("gradcam", "features", None, None, 0.0, None, {}, 0.0)

    model = DummyCNN()
    cache.set(model, 1, config, "hash1", res)

    # Hit
    cached = cache.get(model, 1, config, "hash1")
    assert cached is not None
    assert cached.algorithm == "gradcam"

    # Miss
    miss = cache.get(model, 1, config, "hash2")
    assert miss is None
