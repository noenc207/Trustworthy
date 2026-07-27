from pathlib import Path

files = {}

# 1. src/modules/inference_engine/types.py
files["src/modules/inference_engine/types.py"] = """
\"\"\"
Data Transfer Objects (DTOs) and common types for AI modules.
\"\"\"
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Backend-agnostic tensor type alias
TensorLike = Any


@dataclass
class QualityReport:
    \"\"\"Image quality assessment result.\"\"\"
    overall_score: float
    is_acceptable: bool
    blur_score: float = 0.0
    exposure_score: float = 0.0
    color_score: float = 0.0
    resolution_score: float = 0.0
    artifact_score: float = 0.0
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    \"\"\"Classification output from any classifier.\"\"\"
    logits: TensorLike
    probabilities: TensorLike
    predicted_class: int
    confidence: float
    class_labels: list[str]


@dataclass
class OODResult:
    \"\"\"Out-of-Distribution detection result.\"\"\"
    is_ood: bool
    ood_score: float
    method_name: str
    threshold: float
    confidence: float


@dataclass
class UncertaintyResult:
    \"\"\"Uncertainty estimation result.\"\"\"
    mean_probabilities: Any
    predictive_entropy: float
    mutual_information: float
    aleatoric_variance: float | Any
    epistemic_variance: float | Any
    num_samples: int


@dataclass
class ExplanationResult:
    \"\"\"Explainability output (heatmap / saliency map).\"\"\"
    heatmap: Any
    visualization: Any | None
    target_class: int
    method_name: str


@dataclass
class ClinicalRecommendation:
    \"\"\"Structured clinical recommendation.\"\"\"
    predicted_diagnosis: str
    urgency_level: str
    confidence_level: str
    recommendation_text: str
    patient_summary: str
    next_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_version: str = "unknown"
    disclaimer: str = (
        "This AI-generated recommendation is a decision support tool only. "
        "It does not constitute a medical diagnosis. "
        "Always consult a qualified dermatologist."
    )
"""

# 2. src/modules/inference_engine/protocols.py
files["src/modules/inference_engine/protocols.py"] = """
\"\"\"
AI Module Protocol Interfaces.

Defines structural typing contracts (PEP 544) for every AI module.
Any class that matches a protocol's method signatures is automatically
compatible — no inheritance required.
\"\"\"
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.modules.inference_engine.types import (
    ClassificationResult,
    ClinicalRecommendation,
    ExplanationResult,
    OODResult,
    QualityReport,
    TensorLike,
    UncertaintyResult,
)


@runtime_checkable
class ImageQualityAssessorProtocol(Protocol):
    \"\"\"Contract for image quality assessment modules.\"\"\"

    def assess(self, image: Any) -> QualityReport:
        \"\"\"Assess the quality of a raw image.\"\"\"
        ...


@runtime_checkable
class PreprocessorProtocol(Protocol):
    \"\"\"Contract for preprocessing pipeline modules.\"\"\"

    def __call__(self, image: Any) -> TensorLike:
        \"\"\"Transform a raw image into a normalized model-input tensor.\"\"\"
        ...


@runtime_checkable
class ClassifierProtocol(Protocol):
    \"\"\"Contract for classification modules.\"\"\"

    def predict(self, tensor: TensorLike) -> ClassificationResult:
        \"\"\"Run classification on a preprocessed tensor.\"\"\"
        ...

    def forward(self, x: TensorLike) -> TensorLike:
        \"\"\"Raw forward pass returning logits.\"\"\"
        ...


@runtime_checkable
class OODDetectorProtocol(Protocol):
    \"\"\"Contract for OOD detection modules.\"\"\"

    def detect(self, tensor: TensorLike) -> OODResult:
        \"\"\"Run OOD detection on a preprocessed input tensor.\"\"\"
        ...


@runtime_checkable
class UncertaintyEstimatorProtocol(Protocol):
    \"\"\"Contract for uncertainty estimation modules.\"\"\"

    def estimate(self, tensor: TensorLike) -> UncertaintyResult:
        \"\"\"Estimate uncertainty via stochastic forward passes or ensembles.\"\"\"
        ...


@runtime_checkable
class CalibratorProtocol(Protocol):
    \"\"\"Contract for confidence calibration modules.\"\"\"

    def forward(self, logits: TensorLike) -> TensorLike:
        \"\"\"Return calibrated logits.\"\"\"
        ...


@runtime_checkable
class ExplainerProtocol(Protocol):
    \"\"\"Contract for explainability modules.\"\"\"

    def explain(
        self,
        input_tensor: TensorLike,
        original_image: Any,
        target_class: int | None = None,
    ) -> ExplanationResult:
        \"\"\"Generate a visual explanation for a prediction.\"\"\"
        ...


@runtime_checkable
class RecommendationEngineProtocol(Protocol):
    \"\"\"Contract for clinical recommendation engines.\"\"\"

    def generate(
        self,
        predicted_class: Any,
        confidence: float,
        uncertainty: float,
        is_ood: bool,
        model_version: str = "unknown",
    ) -> ClinicalRecommendation:
        \"\"\"Generate a structured clinical recommendation.\"\"\"
        ...
"""

# 3. src/framework/interfaces/base_model.py
files["src/framework/interfaces/base_model.py"] = """
\"\"\"
Abstract base interfaces for trainable models.
\"\"\"
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AbstractModel(ABC):
    \"\"\"
    Base interface for all inference models.
    Backend-agnostic (no direct PyTorch dependency).
    \"\"\"

    @property
    @abstractmethod
    def model_name(self) -> str:
        \"\"\"Return the architecture or model identifier.\"\"\"
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        \"\"\"Return the dimension of extracted features, if applicable.\"\"\"
        ...

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        \"\"\"Run inference on preprocessed inputs.\"\"\"
        ...

    @abstractmethod
    def load_weights(self, path: Path) -> None:
        \"\"\"Load model weights from the given path.\"\"\"
        ...

    def to_device(self, device_str: str) -> "AbstractModel":
        \"\"\"
        Move the model to the specified device.
        Default implementation is a no-op; subclasses should override.
        \"\"\"
        return self

    def get_info(self) -> dict[str, Any]:
        \"\"\"Return model metadata.\"\"\"
        return {
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
        }
"""

# 4. src/framework/interfaces/base_classifier.py
files["src/framework/interfaces/base_classifier.py"] = """
\"\"\"Abstract interface for classification models.\"\"\"
from __future__ import annotations

from abc import abstractmethod

from src.framework.interfaces.base_model import AbstractModel


class AbstractClassifier(AbstractModel):
    \"\"\"Base interface for classification specific models.\"\"\"

    @property
    @abstractmethod
    def num_classes(self) -> int:
        \"\"\"Return the number of classes this model predicts.\"\"\"
        ...
"""

# 5. src/framework/interfaces/base_segmentor.py
files["src/framework/interfaces/base_segmentor.py"] = """
\"\"\"Abstract interface for segmentation models.\"\"\"
from __future__ import annotations

from abc import abstractmethod

from src.framework.interfaces.base_model import AbstractModel


class AbstractSegmentor(AbstractModel):
    \"\"\"Base interface for segmentation specific models.\"\"\"

    @property
    @abstractmethod
    def num_classes(self) -> int:
        \"\"\"Return the number of segmentation classes.\"\"\"
        ...
"""

# 6. src/framework/interfaces/base_ood.py
files["src/framework/interfaces/base_ood.py"] = """
\"\"\"Abstract interface for Out-of-Distribution detectors.\"\"\"
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractOODDetector(ABC):
    \"\"\"Base interface for OOD detection modules.\"\"\"

    @abstractmethod
    def detect(self, inputs: Any) -> Any:
        \"\"\"Compute OOD score and boolean flag for inputs.\"\"\"
        ...
"""

# 7. src/framework/interfaces/base_calibrator.py
files["src/framework/interfaces/base_calibrator.py"] = """
\"\"\"Abstract interface for confidence calibrators.\"\"\"
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractCalibrator(ABC):
    \"\"\"Base interface for probability calibration modules.\"\"\"

    @abstractmethod
    def forward(self, logits: Any) -> Any:
        \"\"\"Apply calibration to raw model logits.\"\"\"
        ...
"""

# 8. src/framework/interfaces/base_uncertainty.py
files["src/framework/interfaces/base_uncertainty.py"] = """
\"\"\"Abstract interface for uncertainty estimators.\"\"\"
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractUncertaintyEstimator(ABC):
    \"\"\"Base interface for uncertainty estimation modules.\"\"\"

    @abstractmethod
    def estimate(self, inputs: Any) -> Any:
        \"\"\"Estimate uncertainty metrics for inputs.\"\"\"
        ...
"""

# 9. src/framework/interfaces/base_explainer.py
files["src/framework/interfaces/base_explainer.py"] = """
\"\"\"Abstract interface for model explainers.\"\"\"
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractExplainer(ABC):
    \"\"\"Base interface for explainability modules.\"\"\"

    @abstractmethod
    def explain(self, inputs: Any, original: Any, target: int | None = None) -> Any:
        \"\"\"Generate explanations (e.g., heatmaps) for model predictions.\"\"\"
        ...
"""

# 10. src/framework/interfaces/base_quality.py
files["src/framework/interfaces/base_quality.py"] = """
\"\"\"Abstract interface for image quality assessors.\"\"\"
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractQualityAssessor(ABC):
    \"\"\"Base interface for image quality assessment modules.\"\"\"

    @abstractmethod
    def assess(self, image: Any) -> Any:
        \"\"\"Assess the quality of an input image.\"\"\"
        ...
"""

# 11. src/framework/interfaces/base_recommender.py
files["src/framework/interfaces/base_recommender.py"] = """
\"\"\"Abstract interface for clinical recommendation engines.\"\"\"
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractRecommendationEngine(ABC):
    \"\"\"Base interface for clinical recommendation engines.\"\"\"

    @abstractmethod
    def generate(self, **kwargs: Any) -> Any:
        \"\"\"Generate clinical recommendations based on inference results.\"\"\"
        ...
"""

# 12. src/framework/interfaces/__init__.py
files["src/framework/interfaces/__init__.py"] = """
\"\"\"Framework abstract interfaces.\"\"\"
from __future__ import annotations

from .base_model import AbstractModel
from .base_classifier import AbstractClassifier
from .base_segmentor import AbstractSegmentor
from .base_ood import AbstractOODDetector
from .base_calibrator import AbstractCalibrator
from .base_uncertainty import AbstractUncertaintyEstimator
from .base_explainer import AbstractExplainer
from .base_quality import AbstractQualityAssessor
from .base_recommender import AbstractRecommendationEngine

__all__ = [
    "AbstractModel",
    "AbstractClassifier",
    "AbstractSegmentor",
    "AbstractOODDetector",
    "AbstractCalibrator",
    "AbstractUncertaintyEstimator",
    "AbstractExplainer",
    "AbstractQualityAssessor",
    "AbstractRecommendationEngine",
]
"""

# 13. src/framework/metrics/execution.py
files["src/framework/metrics/execution.py"] = """
\"\"\"
Execution metrics for pipeline observability.
Includes timing and resource tracking.
\"\"\"
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TimingMetrics:
    \"\"\"Records duration of individual stages and overall pipeline.\"\"\"
    stage_timings: dict[str, float]
    total_latency_seconds: float
    stage_count: int
    skipped_stages: list[str]
    error_stages: list[str]

    @property
    def slowest_stage(self) -> tuple[str, float] | None:
        \"\"\"Identify the stage that took the most time.\"\"\"
        if not self.stage_timings:
            return None
        stage_name = max(self.stage_timings, key=lambda k: self.stage_timings[k])
        return stage_name, self.stage_timings[stage_name]

    @classmethod
    def from_timing_records(cls, records: list[Any]) -> "TimingMetrics":
        \"\"\"
        Build TimingMetrics from a list of StageTimingRecord instances.
        \"\"\"
        timings = {}
        skipped = []
        errors = []
        total_time = 0.0

        for r in records:
            if r.skipped:
                skipped.append(r.stage_name)
            else:
                timings[r.stage_name] = r.duration_seconds
                total_time += r.duration_seconds
            if r.error is not None:
                errors.append(r.stage_name)

        return cls(
            stage_timings=timings,
            total_latency_seconds=total_time,
            stage_count=len(records),
            skipped_stages=skipped,
            error_stages=errors,
        )


@dataclass
class ResourceMetrics:
    \"\"\"Records resource utilization during execution.\"\"\"
    peak_memory_mb: float | None
    gpu_memory_mb: float | None
    device_str: str

    @classmethod
    def capture_current(cls, device_str: str) -> "ResourceMetrics":
        \"\"\"Capture current resource usage.\"\"\"
        return cls(
            peak_memory_mb=None,
            gpu_memory_mb=None,
            device_str=device_str,
        )


@dataclass
class ExecutionMetadata:
    \"\"\"Comprehensive observability metadata for a pipeline run.\"\"\"
    request_id: str
    model_version: str
    pipeline_version: str
    device_str: str
    class_labels: list[str]
    timestamp_utc: str
    timing: TimingMetrics
    resources: ResourceMetrics
"""

# 14. src/framework/metrics/__init__.py
files["src/framework/metrics/__init__.py"] = """
\"\"\"Framework execution metrics.\"\"\"
from __future__ import annotations

from .execution import TimingMetrics, ResourceMetrics, ExecutionMetadata

__all__ = [
    "TimingMetrics",
    "ResourceMetrics",
    "ExecutionMetadata",
]
"""

# 15. src/framework/common/results.py
files["src/framework/common/results.py"] = """
\"\"\"
Unified pipeline result DTO.
Replaces all previous PredictionResult variants.
\"\"\"
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING

from src.framework.metrics.execution import ExecutionMetadata

# Prevent circular imports while maintaining strict typing
if TYPE_CHECKING:
    from src.modules.inference_engine.types import (
        ClinicalRecommendation,
        ExplanationResult,
        OODResult,
        QualityReport,
        UncertaintyResult,
    )


@dataclass
class PipelineResult:
    \"\"\"Unified response object encapsulating all inference pipeline outputs.\"\"\"
    
    # Identity
    request_id: str
    model_version: str
    pipeline_version: str
    
    # Core prediction
    predicted_class: str
    predicted_class_index: int
    class_probabilities: dict[str, float]
    raw_confidence: float
    calibrated_confidence: float
    
    # Trustworthiness signals
    quality: "QualityReport" | None
    ood: "OODResult" | None
    uncertainty: "UncertaintyResult" | None
    
    # Explainability
    explanation: "ExplanationResult" | None
    
    # Clinical
    recommendation: "ClinicalRecommendation" | None
    
    # Execution metadata
    execution: ExecutionMetadata
    
    # Observability
    warnings: list[str]
    errors: list[str]

    def is_trustworthy(self, ood_threshold: float = 0.5, uncertainty_threshold: float = 0.5) -> bool:
        \"\"\"Determine if the prediction can be fully trusted.\"\"\"
        if self.quality and not self.quality.is_acceptable:
            return False
        if self.ood and self.ood.is_ood:
            return False
        if self.uncertainty and self.uncertainty.predictive_entropy > uncertainty_threshold:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        \"\"\"Convert result to a JSON-serializable dictionary summary.\"\"\"
        return asdict(self)

    def has_errors(self) -> bool:
        \"\"\"Return True if any errors occurred during pipeline execution.\"\"\"
        return len(self.errors) > 0
"""

# 16. src/framework/common/__init__.py
files["src/framework/common/__init__.py"] = """
\"\"\"Framework common types and utilities.\"\"\"
from __future__ import annotations

from .results import PipelineResult

__all__ = [
    "PipelineResult",
]
"""

# 17. src/framework/__init__.py
files["src/framework/__init__.py"] = """
\"\"\"
TrustDerm AI Core Framework.
Provides base interfaces, metrics, and common types for AI pipelines.
\"\"\"
from __future__ import annotations

from .interfaces import (
    AbstractModel,
    AbstractClassifier,
    AbstractSegmentor,
    AbstractOODDetector,
    AbstractCalibrator,
    AbstractUncertaintyEstimator,
    AbstractExplainer,
    AbstractQualityAssessor,
    AbstractRecommendationEngine,
)
from .common import PipelineResult
from .metrics import TimingMetrics, ResourceMetrics, ExecutionMetadata

__all__ = [
    "AbstractModel",
    "AbstractClassifier",
    "AbstractSegmentor",
    "AbstractOODDetector",
    "AbstractCalibrator",
    "AbstractUncertaintyEstimator",
    "AbstractExplainer",
    "AbstractQualityAssessor",
    "AbstractRecommendationEngine",
    "PipelineResult",
    "TimingMetrics",
    "ResourceMetrics",
    "ExecutionMetadata",
]
"""

def write_files():
    base_dir = Path("d:/Trustworthy")
    for file_path, content in files.items():
        full_path = base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip leading newlines to make it clean
        full_path.write_text(content.lstrip(), encoding="utf-8")
        print(f"Created: {file_path}")

if __name__ == "__main__":
    write_files()
