import os
import json
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    optimizer: str
    iterations: int
    training_time: float
    
    ece_before: float
    ece_after: float
    adaptive_ece_before: float
    adaptive_ece_after: float
    mce_before: float
    mce_after: float
    brier_before: float
    brier_after: float
    nll_before: float
    nll_after: float
    
    confidence_before: float
    confidence_after: float
    
    improved: bool
    rollback: bool
    is_calibrated: bool
    execution_time: float
    
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self, filepath: str) -> None:
        """Save a scientific report to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report = {
            "optimizer": self.optimizer,
            "temperature": self.temperature,
            "training_iterations": self.iterations,
            "training_time_sec": self.training_time,
            "execution_time_sec": self.execution_time,
            "rollback": self.rollback,
            "improved": self.improved,
            "status": "CALIBRATED" if self.is_calibrated else "FAILED",
            "metrics": {
                "ece_before": self.ece_before,
                "ece_after": self.ece_after,
                "adaptive_ece_before": self.adaptive_ece_before,
                "adaptive_ece_after": self.adaptive_ece_after,
                "mce_before": self.mce_before,
                "mce_after": self.mce_after,
                "brier_before": self.brier_before,
                "brier_after": self.brier_after,
                "nll_before": self.nll_before,
                "nll_after": self.nll_after,
                "confidence_before": self.confidence_before,
                "confidence_after": self.confidence_after
            },
            "additional_metrics": self.metrics,
            "dataset_statistics": self.metadata.get("dataset_statistics", {}),
            "warnings": self.warnings
        }
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=4)
