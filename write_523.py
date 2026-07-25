import os
from pathlib import Path
import textwrap

files = {}

# 1. src/modules/inference_engine/runtime_state.py
files["src/modules/inference_engine/runtime_state.py"] = """
\"\"\"
Pipeline Runtime State Management.
\"\"\"
from __future__ import annotations
import threading
from enum import Enum

class RuntimeState(str, Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateTransitionError(Exception):
    \"\"\"Raised when an invalid runtime state transition is attempted.\"\"\"
    pass


class RuntimeStateManager:
    \"\"\"Thread-safe runtime state manager with transition validation.\"\"\"
    
    def __init__(self) -> None:
        self._state = RuntimeState.CREATED
        self._lock = threading.RLock()
        self._allowed_transitions: dict[RuntimeState, set[RuntimeState]] = {
            RuntimeState.CREATED: {RuntimeState.INITIALIZING, RuntimeState.CANCELLED},
            RuntimeState.INITIALIZING: {RuntimeState.READY, RuntimeState.FAILED, RuntimeState.CANCELLED},
            RuntimeState.READY: {RuntimeState.RUNNING, RuntimeState.CANCELLED},
            RuntimeState.RUNNING: {RuntimeState.COMPLETED, RuntimeState.FAILED, RuntimeState.CANCELLED},
            RuntimeState.COMPLETED: set(),
            RuntimeState.FAILED: set(),
            RuntimeState.CANCELLED: set(),
        }

    @property
    def current(self) -> RuntimeState:
        with self._lock:
            return self._state

    def transition_to(self, new_state: RuntimeState) -> None:
        with self._lock:
            if new_state not in self._allowed_transitions[self._state]:
                raise StateTransitionError(
                    f"Invalid transition from {self._state.value} to {new_state.value}"
                )
            self._state = new_state
"""

# 2. src/modules/inference_engine/context.py
files["src/modules/inference_engine/context.py"] = """
\"\"\"
Pipeline execution context and configuration.
\"\"\"
from __future__ import annotations
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.modules.inference_engine.device_manager import DeviceManager


@dataclass(frozen=True)
class PipelineConfig:
    \"\"\"Immutable configuration snapshot.\"\"\"
    model_version: str = "v1"
    pipeline_version: str = "5.2"
    device_str: str = "cpu"
    class_labels: list[str] = field(default_factory=list)
    generate_explanation: bool = True
    ood_method: str = "energy"
    uncertainty_samples: int = 30
    uncertainty_samples_max: int = 100


class CancellationToken:
    \"\"\"Thread-safe cancellation token.\"\"\"
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
    \"\"\"Runtime execution context serving as the Blackboard for the pipeline.\"\"\"
    raw_image: Any
    config: PipelineConfig
    
    # Identity and Tracing
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Runtime Metadata
    device_manager: "DeviceManager | None" = None
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
    
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        
    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
"""

# 3. src/modules/inference_engine/stages.py
files["src/modules/inference_engine/stages.py"] = """
\"\"\"
Reusable execution stage abstraction.
\"\"\"
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.modules.inference_engine.context import PipelineContext


@dataclass
class StagePolicy:
    \"\"\"Execution policy for a stage.\"\"\"
    timeout_seconds: float = 30.0
    retry_count: int = 0
    skip_on_failure: bool = False
    is_critical: bool = True


class PipelineStage(ABC):
    \"\"\"
    Abstract Base Class for a pipeline execution stage.
    Backend-agnostic, communicates strictly through PipelineContext.
    \"\"\"
    
    def __init__(
        self, 
        name: str, 
        policy: StagePolicy | None = None, 
        dependencies: list[str] | None = None
    ) -> None:
        self.name = name
        self.policy = policy or StagePolicy()
        self.dependencies = dependencies or []

    @abstractmethod
    def initialize(self) -> None:
        \"\"\"Setup resources before execution starts.\"\"\"
        pass

    @abstractmethod
    def validate(self, context: PipelineContext) -> bool:
        \"\"\"Check if the stage should execute based on current context.\"\"\"
        pass

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        \"\"\"Perform the main stage logic and mutate the context.\"\"\"
        pass

    @abstractmethod
    def cleanup(self) -> None:
        \"\"\"Release resources after execution completes or fails.\"\"\"
        pass
"""

# 4. src/modules/inference_engine/graph.py
files["src/modules/inference_engine/graph.py"] = """
\"\"\"
Immutable execution graph for pipeline stages.
\"\"\"
from __future__ import annotations
from typing import Sequence

from src.modules.inference_engine.stages import PipelineStage


class ExecutionGraph:
    \"\"\"
    Manages stage ordering and topological verification.
    \"\"\"
    
    def __init__(self, stages: Sequence[PipelineStage]) -> None:
        self._stages = tuple(stages)  # immutable sequence
        self._verify_topology()

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        return self._stages

    def _verify_topology(self) -> None:
        \"\"\"
        Validate dependencies and detect cycles for a sequential pipeline.
        Since execution is sequential in the provided order, any dependency 
        must appear before the stage that depends on it.
        \"\"\"
        stage_names = {s.name for s in self._stages}
        
        seen = set()
        for stage in self._stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    raise ValueError(f"Stage '{stage.name}' has missing dependency '{dep}'")
                if dep not in seen:
                    raise ValueError(
                        f"Topological error: Stage '{stage.name}' depends on '{dep}', "
                        f"which is not executed before it."
                    )
            seen.add(stage.name)
"""

# 5. src/modules/inference_engine/lifecycle.py
files["src/modules/inference_engine/lifecycle.py"] = """
\"\"\"
Pipeline lifecycle hooks.
\"\"\"
from __future__ import annotations
from typing import Callable

from src.modules.inference_engine.context import PipelineContext
from src.modules.inference_engine.stages import PipelineStage


class PipelineLifecycle:
    \"\"\"Registry for runtime lifecycle hook callbacks.\"\"\"
    
    def __init__(self) -> None:
        self.before_pipeline: list[Callable[[PipelineContext], None]] = []
        self.after_pipeline: list[Callable[[PipelineContext], None]] = []
        
        self.before_stage: list[Callable[[PipelineStage, PipelineContext], None]] = []
        self.after_stage: list[Callable[[PipelineStage, PipelineContext], None]] = []
        
        self.before_cleanup: list[Callable[[PipelineContext], None]] = []
        self.after_cleanup: list[Callable[[PipelineContext], None]] = []
        
        self.on_failure: list[Callable[[PipelineContext, Exception], None]] = []
        self.on_cancel: list[Callable[[PipelineContext], None]] = []
"""

# 6. src/modules/inference_engine/validator.py
files["src/modules/inference_engine/validator.py"] = """
\"\"\"
Runtime consistency validator.
\"\"\"
from __future__ import annotations
from typing import Sequence

from src.modules.inference_engine.stages import PipelineStage
from src.modules.inference_engine.context import PipelineConfig


class RuntimeValidator:
    \"\"\"Validates pipeline and configuration integrity.\"\"\"
    
    @staticmethod
    def validate_stages(stages: Sequence[PipelineStage]) -> None:
        \"\"\"Check for duplicate stages.\"\"\"
        seen = set()
        for s in stages:
            if s.name in seen:
                raise ValueError(f"Duplicate stage name found: {s.name}")
            seen.add(s.name)
            
    @staticmethod
    def validate_config(config: PipelineConfig | None) -> None:
        \"\"\"Ensure configuration is present and valid.\"\"\"
        if config is None:
            raise ValueError("Configuration snapshot is missing.")
        if not config.device_str:
            raise ValueError("Device string is missing from configuration.")
"""

# 7. src/modules/inference_engine/executor.py
files["src/modules/inference_engine/executor.py"] = """
\"\"\"
Main execution engine for inference pipelines.
\"\"\"
from __future__ import annotations
import time
import logging
import concurrent.futures
from typing import Any, Sequence

from src.modules.inference_engine.context import PipelineContext, PipelineConfig
from src.modules.inference_engine.stages import PipelineStage
from src.modules.inference_engine.graph import ExecutionGraph
from src.modules.inference_engine.runtime_state import RuntimeStateManager, RuntimeState
from src.modules.inference_engine.lifecycle import PipelineLifecycle
from src.modules.inference_engine.validator import RuntimeValidator
from src.modules.inference_engine.events import EventDispatcher, EventType, EventPayload
from src.framework.common.results import PipelineResult
from src.framework.metrics.execution import TimingMetrics, ResourceMetrics, ExecutionMetadata
from src.modules.inference_engine.device_manager import DeviceManager

logger = logging.getLogger(__name__)


class PipelineExecutor:
    \"\"\"
    Runtime executor for AI inference pipelines.
    Handles sequential execution, timeout, recovery, and telemetry.
    \"\"\"

    def __init__(
        self, 
        stages: Sequence[PipelineStage], 
        dispatcher: EventDispatcher | None = None
    ) -> None:
        self._dispatcher = dispatcher or EventDispatcher()
        self._lifecycle = PipelineLifecycle()
        
        # Validation
        RuntimeValidator.validate_stages(stages)
        self._graph = ExecutionGraph(stages)
        
        self._state_manager = RuntimeStateManager()
        self._state_manager.transition_to(RuntimeState.INITIALIZING)
        
        for stage in self._graph.stages:
            stage.initialize()
            
        self._state_manager.transition_to(RuntimeState.READY)

    @property
    def lifecycle(self) -> PipelineLifecycle:
        return self._lifecycle

    @property
    def state(self) -> RuntimeState:
        return self._state_manager.current

    def execute(self, raw_image: Any, config: PipelineConfig) -> PipelineResult:
        \"\"\"Execute the pipeline sequentially and produce a unified result.\"\"\"
        RuntimeValidator.validate_config(config)
        
        context = PipelineContext(raw_image=raw_image, config=config)
        context.device_manager = DeviceManager(config.device_str)
        
        self._state_manager.transition_to(RuntimeState.RUNNING)
        
        stage_timings: dict[str, float] = {}
        skipped: list[str] = []
        errors: list[str] = []
        
        try:
            self._run_lifecycle_hooks(self._lifecycle.before_pipeline, context)
            self._dispatcher.emit(EventPayload(
                EventType.BEFORE_PREDICTION, context.request_id, str(time.time()), 
                extra={"execution_id": context.execution_id}
            ))

            for stage in self._graph.stages:
                if context.cancellation_token.is_cancelled:
                    self._handle_cancel(context)
                    break
                    
                if not stage.validate(context):
                    skipped.append(stage.name)
                    self._dispatcher.emit(EventPayload(
                        EventType.STAGE_SKIPPED, context.request_id, str(time.time()), 
                        stage_name=stage.name
                    ))
                    continue

                self._run_lifecycle_hooks(self._lifecycle.before_stage, stage, context)
                self._dispatcher.emit(EventPayload(
                    EventType.BEFORE_STAGE, context.request_id, str(time.time()), 
                    stage_name=stage.name
                ))

                start_time = time.time()
                try:
                    context = self._execute_stage_with_retry(stage, context)
                except Exception as e:
                    errors.append(stage.name)
                    logger.error(f"Stage {stage.name} failed: {e}")
                    context.add_error(f"{stage.name}: {str(e)}")
                    self._dispatcher.emit(EventPayload(
                        EventType.STAGE_ERROR, context.request_id, str(time.time()), 
                        stage_name=stage.name, error=e
                    ))
                    
                    if stage.policy.is_critical and not stage.policy.skip_on_failure:
                        raise e  # Fail the pipeline

                duration = time.time() - start_time
                stage_timings[stage.name] = duration

                self._run_lifecycle_hooks(self._lifecycle.after_stage, stage, context)
                if stage.name not in errors:
                    self._dispatcher.emit(EventPayload(
                        EventType.AFTER_STAGE, context.request_id, str(time.time()), 
                        stage_name=stage.name, extra={"duration": duration}
                    ))

            if self.state == RuntimeState.RUNNING:
                self._state_manager.transition_to(RuntimeState.COMPLETED)
            
        except Exception as e:
            if self.state != RuntimeState.CANCELLED:
                self._state_manager.transition_to(RuntimeState.FAILED)
            self._run_lifecycle_hooks(self._lifecycle.on_failure, context, e)
            self._dispatcher.emit(EventPayload(
                EventType.PIPELINE_ERROR, context.request_id, str(time.time()), error=e
            ))
            context.add_error(f"Pipeline failed: {str(e)}")
            
        finally:
            self._run_lifecycle_hooks(self._lifecycle.before_cleanup, context)
            for stage in self._graph.stages:
                stage.cleanup()
            self._run_lifecycle_hooks(self._lifecycle.after_cleanup, context)
            self._run_lifecycle_hooks(self._lifecycle.after_pipeline, context)
            self._dispatcher.emit(EventPayload(
                EventType.AFTER_PREDICTION, context.request_id, str(time.time())
            ))

        return self._build_result(context, stage_timings, skipped, errors)

    def _execute_stage_with_retry(self, stage: PipelineStage, context: PipelineContext) -> PipelineContext:
        \"\"\"Executes a stage with timeout and retry support.\"\"\"
        attempts = 0
        max_attempts = stage.policy.retry_count + 1
        
        while attempts < max_attempts:
            try:
                attempts += 1
                return self._execute_stage_with_timeout(stage, context)
            except Exception as e:
                if attempts >= max_attempts:
                    raise e
                logger.warning(f"Retrying stage {stage.name} (attempt {attempts}/{max_attempts}) due to {e}")
                
        return context

    def _execute_stage_with_timeout(self, stage: PipelineStage, context: PipelineContext) -> PipelineContext:
        \"\"\"Executes a stage with a hard timeout using a thread pool.\"\"\"
        if stage.policy.timeout_seconds <= 0:
            return stage.execute(context)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(stage.execute, context)
            try:
                return future.result(timeout=stage.policy.timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Stage '{stage.name}' exceeded timeout of {stage.policy.timeout_seconds}s")

    def _handle_cancel(self, context: PipelineContext) -> None:
        \"\"\"Handle a cancellation token signal.\"\"\"
        logger.info(f"Execution {context.execution_id} cancelled.")
        self._state_manager.transition_to(RuntimeState.CANCELLED)
        context.add_warning("Pipeline execution cancelled via token.")
        self._run_lifecycle_hooks(self._lifecycle.on_cancel, context)

    def _run_lifecycle_hooks(self, hooks: list, *args: Any) -> None:
        \"\"\"Run a list of lifecycle hooks safely.\"\"\"
        for hook in hooks:
            try:
                hook(*args)
            except Exception as e:
                logger.warning(f"Lifecycle hook failed: {e}")

    def _build_result(
        self, context: PipelineContext, timings: dict[str, float], 
        skipped: list[str], errors: list[str]
    ) -> PipelineResult:
        \"\"\"Construct the unified PipelineResult from Context.\"\"\"
        tm = TimingMetrics(
            stage_timings=timings,
            total_latency_seconds=sum(timings.values()),
            stage_count=len(self._graph.stages),
            skipped_stages=skipped,
            error_stages=errors
        )
        rm = ResourceMetrics.capture_current(context.config.device_str)
        meta = ExecutionMetadata(
            request_id=context.request_id,
            model_version=context.config.model_version,
            pipeline_version=context.config.pipeline_version,
            device_str=context.config.device_str,
            class_labels=context.config.class_labels,
            timestamp_utc=str(context.created_at_utc),
            timing=tm,
            resources=rm
        )
        
        # Safely extract from ClassificationResult and others if present,
        # but PipelineContext holds the domain typed objects directly.
        c_res = context.classification
        
        return PipelineResult(
            request_id=context.request_id,
            model_version=context.config.model_version,
            pipeline_version=context.config.pipeline_version,
            predicted_class=c_res.class_labels[c_res.predicted_class] if c_res else "unknown",
            predicted_class_index=c_res.predicted_class if c_res else -1,
            class_probabilities=dict(zip(c_res.class_labels, c_res.probabilities)) if c_res else {},
            raw_confidence=c_res.confidence if c_res else 0.0,
            calibrated_confidence=context.calibrated_confidence if context.calibrated_confidence else 0.0,
            quality=context.quality,
            ood=context.ood,
            uncertainty=context.uncertainty,
            explanation=context.explanation,
            recommendation=context.recommendation,
            execution=meta,
            warnings=context.warnings,
            errors=context.errors
        )
"""

def write_files():
    base_dir = Path("d:/Trustworthy")
    for file_path, content in files.items():
        full_path = base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content.lstrip(), encoding="utf-8")
        print(f"Created: {file_path}")

if __name__ == "__main__":
    write_files()
