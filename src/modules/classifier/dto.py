from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
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
    probabilities: Dict[str, float]
    confidence: float
    logits: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    runtime_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    status: str = "SUCCESS"

@dataclass(frozen=True)
class PredictionSummary:
    total_predictions: int
    mean_confidence: float
    latency_stats: Dict[str, float]

@dataclass(frozen=True)
class BatchPredictionResult:
    predictions: List[PredictionResult]
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
    class_mapping: Dict[int, str]
    configuration_hash: str
    training_metadata: Dict[str, Any]

@dataclass(frozen=True)
class RuntimeMetadata:
    device: str
    backend: str
    mixed_precision: bool
    batch_size: int
    latency: float
