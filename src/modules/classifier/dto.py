from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionCandidate:
    class_name: str
    class_index: int
    probability: float

@dataclass(frozen=True)
class PredictionResult:
    prediction: str
    class_index: int
    class_name: str
    probabilities: dict[str, float]
    confidence: float
    logits: np.ndarray | None = None
    embedding: np.ndarray | None = None
    runtime_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    status: str = "SUCCESS"

@dataclass(frozen=True)
class PredictionSummary:
    total_predictions: int
    mean_confidence: float
    latency_stats: dict[str, float]

@dataclass(frozen=True)
class BatchPredictionResult:
    predictions: list[PredictionResult]
    summary: PredictionSummary
    batch_runtime_ms: float
    valid: bool = True
    status: str = "SUCCESS"

@dataclass(frozen=True)
class ModelMetadata:
    architecture: str
    backbone: str
    classifier_type: str
    parameter_count: int
    embedding_dimension: int
    num_classes: int
    input_shape: tuple

@dataclass(frozen=True)
class CheckpointMetadata:
    repository_version: str
    framework_version: str
    torch_version: str
    python_version: str
    creation_timestamp: str
    git_commit_hash: str
    class_mapping: dict[int, str]
    configuration_hash: str
    training_metadata: dict[str, Any]

@dataclass(frozen=True)
class RuntimeMetadata:
    device: str
    backend: str
    mixed_precision: bool
    batch_size: int
    latency: float
