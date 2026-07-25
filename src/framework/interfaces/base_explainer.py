"""Abstract interface for model explainers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractExplainer(ABC):
    """Base interface for explainability modules."""

    @abstractmethod
    def explain(self, inputs: Any, original: Any, target: int | None = None) -> Any:
        """Generate explanations (e.g., heatmaps) for model predictions."""
        ...
