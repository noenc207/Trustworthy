from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ModelInfo:
    model_name: str
    model_version: str
    backend_name: str
    input_shape: tuple[int, ...]
    output_classes: int
    checksum: str
    parameter_count: int
    precision: str
    quantized: bool
    supports_batch: bool
    supports_gradcam: bool
    supports_uncertainty: bool
    supports_fp16: bool

@dataclass(frozen=True)
class InferenceSession:
    request_id: str
    execution_id: str
    backend: str
    device: str
    latency: float
    memory_mb: float
    pipeline_version: str
    preprocessing_version: str
    model_version: str
    batch_size: int
    start_time: float
    end_time: float

@dataclass(frozen=True)
class PredictionCandidate:
    class_name: str
    class_index: int
    probability: float

@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    top_k: list[PredictionCandidate]
    warnings: list[str] = field(default_factory=list)
