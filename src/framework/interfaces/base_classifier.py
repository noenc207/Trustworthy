"""Abstract interface for classification models."""
from __future__ import annotations

from abc import abstractmethod

from src.framework.interfaces.base_model import AbstractModel


class AbstractClassifier(AbstractModel):
    """Base interface for classification specific models."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of classes this model predicts."""
        ...
