"""
Mock Runtime Components for Framework Testing.
"""
from __future__ import annotations

from typing import Any

from src.framework.interfaces import (
    AbstractClassifier,
    AbstractExplainer,
    AbstractOODDetector,
    AbstractQualityAssessor,
    AbstractRecommendationEngine,
)
from src.modules.inference_engine.device_manager import DeviceManager
from src.modules.model_registry.adapter import BackendAdapter


class MockClassifier(AbstractClassifier):
    @property
    def model_name(self) -> str: return "mock_classifier"
    @property
    def feature_dim(self) -> int: return 128
    @property
    def num_classes(self) -> int: return 7
    def load_weights(self, path: Any) -> None: pass
    def predict(self, inputs: Any) -> Any: return {"predicted_class": 1}

class MockOODDetector(AbstractOODDetector):
    def detect(self, inputs: Any) -> Any: return {"is_ood": False, "score": 0.1}

class MockQualityAssessor(AbstractQualityAssessor):
    def assess(self, image: Any) -> Any: return {"is_acceptable": True, "score": 0.9}

class MockExplainer(AbstractExplainer):
    def explain(self, inputs: Any, original: Any, target: int | None = None) -> Any:
        return {"heatmap": "mock_map"}

class MockRecommendationEngine(AbstractRecommendationEngine):
    def generate(self, **kwargs: Any) -> Any:
        return {"recommendation": "Mock recommend"}

class MockBackendAdapter(BackendAdapter):
    def load_model(self, path: Any, device: DeviceManager) -> Any:
        return "mock_model_loaded"
    def to_device(self, model: Any, device: DeviceManager) -> Any:
        return model
    def predict(self, model: Any, inputs: Any) -> Any:
        return {"prediction": "mock"}
