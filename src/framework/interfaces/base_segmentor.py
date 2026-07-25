"""Abstract interface for segmentation models."""
from __future__ import annotations

from abc import abstractmethod

from src.framework.interfaces.base_model import AbstractModel


class AbstractSegmentor(AbstractModel):
    """Base interface for segmentation specific models."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of segmentation classes."""
        ...
