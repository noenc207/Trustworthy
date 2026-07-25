"""
Abstract base interfaces for trainable models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AbstractModel(ABC):
    """
    Base interface for all inference models.
    Backend-agnostic (no direct PyTorch dependency).
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the architecture or model identifier."""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimension of extracted features, if applicable."""
        ...

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Run inference on preprocessed inputs."""
        ...

    @abstractmethod
    def load_weights(self, path: Path) -> None:
        """Load model weights from the given path."""
        ...

    def to_device(self, device_str: str) -> AbstractModel:
        """
        Move the model to the specified device.
        Default implementation is a no-op; subclasses should override.
        """
        return self

    def get_info(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
        }
