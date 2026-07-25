from abc import ABC, abstractmethod
from typing import Any, List
from pathlib import Path
import numpy as np

from src.modules.classifier.result import ModelInfo, PredictionResult

class BackendAdapter(ABC):
    """Abstract interface for ML backends (Torch, ONNX, etc)."""
    @abstractmethod
    def load_model(self, path: Path, device: Any) -> Any:
        pass
        
    @abstractmethod
    def to_device(self, model: Any, device: Any) -> Any:
        pass
        
    @abstractmethod
    def predict(self, model: Any, inputs: Any) -> Any:
        pass

class ClassifierStrategy(ABC):
    """Strategy pattern for model backbone topologies."""
    
    @abstractmethod
    def load(self, weights_path: Path, backend_adapter: BackendAdapter) -> None:
        """Load model weights using backend adapter."""
        pass
        
    @abstractmethod
    def adapt_input(self, image: np.ndarray) -> Any:
        """Convert Numpy image to Backend tensor format."""
        pass
        
    @abstractmethod
    def predict(self, tensor: Any, backend_adapter: BackendAdapter) -> Any:
        """Execute single inference."""
        pass
        
    @abstractmethod
    def predict_batch(self, tensors: List[Any], backend_adapter: BackendAdapter) -> List[PredictionResult]:
        """Execute batch inference."""
        pass
        
    @abstractmethod
    def postprocess_output(self, raw_output: Any) -> dict:
        """Convert backend logits to normalized output dictionary."""
        pass
        
    @abstractmethod
    def model_metadata(self) -> ModelInfo:
        """Return topology specific hardware constraints and info."""
        pass
