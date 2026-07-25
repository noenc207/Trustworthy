"""Abstract interface for clinical recommendation engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractRecommendationEngine(ABC):
    """Base interface for clinical recommendation engines."""

    @abstractmethod
    def generate(self, **kwargs: Any) -> Any:
        """Generate clinical recommendations based on inference results."""
        ...
