from typing import Any
from dataclasses import dataclass, field

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
