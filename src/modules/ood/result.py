from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class OODResult:
    is_in_distribution: bool
    ood_score: float
    confidence: float
    algorithm: str
    threshold: float
    reason: Optional[str]
    execution_time: float
    warnings: list[str] = field(default_factory=list)
