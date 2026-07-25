"""Abstract interface for confidence calibrators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractCalibrator(ABC):
    """Base interface for probability calibration modules."""

    @abstractmethod
    def forward(self, logits: Any) -> Any:
        """Apply calibration to raw model logits."""
        ...
