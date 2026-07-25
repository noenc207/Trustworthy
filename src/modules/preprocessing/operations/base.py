from abc import ABC, abstractmethod

import numpy as np


class AbstractPreprocessingOperation(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the operation."""

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply operation to image."""

    @abstractmethod
    def is_enabled(self) -> bool:
        """Return True if the operation is enabled via config."""
