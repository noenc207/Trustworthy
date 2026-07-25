"""
Backend Adapter Abstraction.
"""
from __future__ import annotations
from typing import Protocol, Any
from pathlib import Path
from src.modules.inference_engine.device_manager import DeviceManager

class BackendAdapter(Protocol):
    """
    Backend abstraction adapter interface.
    Isolates the framework from PyTorch, ONNX, TensorRT, OpenVINO, etc.
    Concrete implementations are responsible for actual library imports.
    """
    def load_model(self, path: Path, device: DeviceManager) -> Any:
        """Load model from path onto target device."""
        ...
        
    def to_device(self, model: Any, device: DeviceManager) -> Any:
        """Move model to a specific device."""
        ...

    def predict(self, model: Any, inputs: Any) -> Any:
        """Run inference."""
        ...
