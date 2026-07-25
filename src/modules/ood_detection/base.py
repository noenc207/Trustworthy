from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from .result import OODResult


class BaseOODDetector(ABC):
    """Base class for all Out-of-Distribution Detectors."""

    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        self.model = model

    @abstractmethod
    def compute_score(self, x: torch.Tensor) -> float:
        """Compute raw OOD score. Higher typically means more OOD."""

    @abstractmethod
    def get_threshold(self) -> float:
        """Get the decision threshold."""

    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the OOD algorithm."""

    def detect(self, x: torch.Tensor) -> OODResult:
        """
        Run detection and package into formal OODResult DTO.
        Numerical stability and exception safeguards are included.
        """
        warnings: list[str] = []

        # Memory profiling
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated()
        else:
            mem_before = 0

        start_time = time.perf_counter()

        # Ensure model is in eval mode
        was_training = self.model.training
        self.model.eval()

        score: float | None = None
        valid = True
        status = "SUCCESS"

        try:
            score = self.compute_score(x)
            if math.isnan(score) or math.isinf(score):
                warnings.append(f"Numerical instability: score is {score}")
                valid = False
                status = "NUMERICAL_INSTABILITY"
                score = None
        except Exception as e:
            warnings.append(f"Computation error: {e!s}")
            valid = False
            status = "COMPUTATION_ERROR"
            score = None

        if was_training:
            self.model.train()

        runtime = time.perf_counter() - start_time
        latency_ms = runtime * 1000.0

        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated()
            memory_mb = max(0.0, float(mem_after - mem_before) / (1024 * 1024))
        else:
            memory_mb = 0.0

        threshold = self.get_threshold()

        if score is not None:
            is_ood = score > threshold
            confidence = max(0.0, min(1.0, 1.0 - score))
        else:
            is_ood = False
            confidence = None

        return OODResult(
            method=self.get_method_name(),
            score=score,
            confidence=confidence,
            uncertainty=score,
            threshold=threshold,
            decision="OUT_OF_DISTRIBUTION" if is_ood else "IN_DISTRIBUTION",
            valid=valid,
            status=status,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            warnings=warnings,
            metadata={}
        )
