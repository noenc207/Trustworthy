"""Abstract interface for image quality assessors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractQualityAssessor(ABC):
    """Base interface for image quality assessment modules."""

    @abstractmethod
    def assess(self, image: Any) -> Any:
        """Assess the quality of an input image."""
        ...
