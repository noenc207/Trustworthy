"""Abstract interface for uncertainty estimators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractUncertaintyEstimator(ABC):
    """Base interface for uncertainty estimation modules."""

    @abstractmethod
    def estimate(self, inputs: Any) -> Any:
        """Estimate uncertainty metrics for inputs."""
        ...
