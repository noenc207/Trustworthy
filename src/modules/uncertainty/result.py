from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UncertaintyResult:
    is_reliable: bool
    uncertainty_score: float
    confidence_interval: tuple[float, float]
    predictive_entropy: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    algorithm: str
    execution_time: float
    threshold: float
    reason: str | None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
