"""
Pipeline execution context and configuration.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.modules.inference_engine.device_manager import DeviceManager


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration snapshot."""
    model_version: str = "v1"
    pipeline_version: str = "5.2"
    device_str: str = "cpu"
    class_labels: list[str] = field(default_factory=list)
    generate_explanation: bool = True
    ood_method: str = "energy"
    uncertainty_samples: int = 30
    uncertainty_samples_max: int = 100


class CancellationToken:
    """Thread-safe cancellation token."""
    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.RLock()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


@dataclass
class PipelineContext:
    """Runtime execution context serving as the Blackboard for the pipeline."""
    raw_image: Any
    config: PipelineConfig

    # Identity and Tracing
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Runtime Metadata
    device_manager: DeviceManager | None = None
    created_at_utc: float = field(default_factory=time.time)
    execution_mode: str = "inference"
    cancellation_token: CancellationToken = field(default_factory=CancellationToken)

    # Shared Runtime State
    tensor: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str | None = None

    # Domain Results
    quality: Any | None = None
    classification: Any | None = None
    ood: Any | None = None
    uncertainty: Any | None = None
    calibrated_probabilities: Any | None = None
    calibrated_confidence: float | None = None
    explanation: Any | None = None
    recommendation: Any | None = None

    # === DERMA-ACT: Active Evidence Acquisition State ===
    # Active perception loop state
    active_state: Any | None = None  # EvidenceState dataclass
    observation_history: list[tuple[int, Any]] = field(default_factory=list)  # (action_idx, features)
    candidate_actions: list[int] = field(default_factory=list)
    action_scores: dict[int, float] = field(default_factory=dict)

    # Evidence map from self-critique
    evidence_map: dict[str, Any] = field(default_factory=dict)
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)

    # Decision quality and safety
    fragility_score: float | None = None
    evidence_budget: int = 3
    stopping_score: float | None = None
    prediction_set: set[int] = field(default_factory=set)
    selective_risk: float | None = None
    selective_decision: str | None = None  # "classify" | "differential" | "review" | "abstain"

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
