from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalibrationResult:
    value: float
    confidence_interval: tuple[float, float]
    method: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_ms: float = 0.0
    valid: bool = True
    status: str = "SUCCESS"
