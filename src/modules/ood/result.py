from dataclasses import dataclass, field


@dataclass(frozen=True)
class OODResult:
    is_in_distribution: bool
    ood_score: float
    confidence: float
    algorithm: str
    threshold: float
    reason: str | None
    execution_time: float
    warnings: list[str] = field(default_factory=list)
