from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.infrastructure.ml_backends.torch.adapter import TorchBackendAdapter
from src.modules.classifier.result import PredictionResult
from src.modules.explainability.config import ExplainabilityConfig
from src.modules.explainability.result import (
    ClinicalReport,
    ConsensusResult,
    ExplainabilityResult,
    MedicalMetrics,
)


class HookManagerInterface(ABC):
    @abstractmethod
    def register_hooks(self, layer_name: str) -> None:
        pass

    @abstractmethod
    def get_activations(self) -> Any:
        pass

    @abstractmethod
    def get_gradients(self) -> Any:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

class LayerResolverInterface(ABC):
    @staticmethod
    @abstractmethod
    def resolve(model: Any) -> str:
        pass

class ExplainerStrategy(ABC):
    """Base strategy for XAI algorithms."""
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        self.raw_heatmap: np.ndarray | None = None
        self.normalized_heatmap: np.ndarray | None = None

    @abstractmethod
    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: HookManagerInterface) -> None:
        pass

    @abstractmethod
    def compute(self, activations: Any, gradients: Any) -> np.ndarray:
        pass

    @abstractmethod
    def normalize(self, raw_heatmap: np.ndarray) -> np.ndarray:
        pass

class EvaluatorInterface(ABC):
    """Base evaluator for Faithfulness, Stability, and Sanity."""
    @abstractmethod
    def evaluate(self, model: Any, image: np.ndarray, tensor: Any, strategy: ExplainerStrategy,
                 prediction: PredictionResult, adapter: TorchBackendAdapter) -> Any:
        pass

class ConsensusEngineInterface(ABC):
    @abstractmethod
    def aggregate(self, heatmaps: list[np.ndarray]) -> ConsensusResult:
        pass

class MedicalMetricsEngineInterface(ABC):
    @abstractmethod
    def compute(self, heatmap: np.ndarray, original_image: np.ndarray, ground_truth_mask: np.ndarray | None = None) -> MedicalMetrics:
        pass

class VisualizationEngineInterface(ABC):
    @abstractmethod
    def generate_all(self, result: ExplainabilityResult, original_image: np.ndarray, output_dir: str) -> None:
        pass

class ReportGeneratorInterface(ABC):
    @abstractmethod
    def generate(self, result: ExplainabilityResult, config: ExplainabilityConfig) -> ClinicalReport:
        pass

    @abstractmethod
    def export(self, report: ClinicalReport, output_dir: str) -> None:
        pass

class ValidatorInterface(ABC):
    @staticmethod
    @abstractmethod
    def validate_tensors(activations: Any, gradients: Any) -> None:
        pass

    @staticmethod
    @abstractmethod
    def validate_heatmap(heatmap: np.ndarray) -> None:
        pass
