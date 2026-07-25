from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OODResult:
    """Out-of-Distribution detection result. (Frozen Architectural DTO)"""
    method: str
    score: float | None
    confidence: float | None
    uncertainty: float | None
    threshold: float | None
    decision: str
    valid: bool
    status: str
    latency_ms: float
    memory_mb: float
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
