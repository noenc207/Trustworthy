"""Abstract interface for Out-of-Distribution detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractOODDetector(ABC):
    """Base interface for OOD detection modules."""

    @abstractmethod
    def detect(self, inputs: Any) -> Any:
        """Compute OOD score and boolean flag for inputs."""
        ...
