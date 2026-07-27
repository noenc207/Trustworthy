from dataclasses import dataclass, field
from typing import Any


@dataclass
class UncertaintyResult:
    uncertainty: float
    confidence: float
    entropy: float
    mutual_information: float
    variation_ratio: float
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_ms: float = 0.0
    valid: bool = True
    status: str = "SUCCESS"
