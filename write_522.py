from pathlib import Path

files = {}

files["src/framework/common/registry.py"] = """
\"\"\"
Thread-safe generic Component Registry.
Replaces all domain-specific registries.
\"\"\"
from __future__ import annotations

import threading
from typing import Callable, Generic, TypeVar, Any

T = TypeVar("T")


class RegistryFrozenError(Exception):
    \"\"\"Raised when attempting to modify a frozen registry.\"\"\"
    pass


class ComponentRegistry(Generic[T]):
    \"\"\"
    Generic, thread-safe, typed component registry.
    Supports lazy instantiation and plugin discovery.
    \"\"\"

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, T | Callable[[], T]] = {}
        self._is_factory: dict[str, bool] = {}
        self._lock = threading.RLock()
        self._frozen = False

    def freeze(self) -> None:
        \"\"\"Freeze the registry to prevent further modifications.\"\"\"
        with self._lock:
            self._frozen = True

    def register(self, key: str, value: T, overwrite: bool = False) -> None:
        \"\"\"Register a concrete instance.\"\"\"
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(f"Registry '{self.name}' is frozen.")
            if not overwrite and key in self._store:
                raise KeyError(f"Key '{key}' already registered in {self.name}.")
            self._store[key] = value
            self._is_factory[key] = False

    def register_factory(self, key: str, factory: Callable[[], T], overwrite: bool = False) -> None:
        \"\"\"Register a factory for lazy instantiation.\"\"\"
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(f"Registry '{self.name}' is frozen.")
            if not overwrite and key in self._store:
                raise KeyError(f"Key '{key}' already registered in {self.name}.")
            self._store[key] = factory
            self._is_factory[key] = True

    def unregister(self, key: str) -> None:
        \"\"\"Remove a component from the registry.\"\"\"
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(f"Registry '{self.name}' is frozen.")
            if key in self._store:
                del self._store[key]
                del self._is_factory[key]

    def contains(self, key: str) -> bool:
        \"\"\"Check if a key exists in the registry.\"\"\"
        with self._lock:
            return key in self._store

    def get(self, key: str) -> T:
        \"\"\"Retrieve a component, instantiating it if registered as a factory.\"\"\"
        with self._lock:
            if key not in self._store:
                raise KeyError(f"Key '{key}' not found in {self.name}.")
            val = self._store[key]
            if self._is_factory[key]:
                # Lazy instantiation
                instance = val()  # type: ignore
                self._store[key] = instance
                self._is_factory[key] = False
                return instance
            return val  # type: ignore

    def list(self) -> list[str]:
        \"\"\"Return all registered keys.\"\"\"
        with self._lock:
            return list(self._store.keys())

    def clear(self) -> None:
        \"\"\"Clear all registrations. Used primarily for testing.\"\"\"
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(f"Registry '{self.name}' is frozen.")
            self._store.clear()
            self._is_factory.clear()
"""

files["src/modules/inference_engine/events.py"] = """
\"\"\"
Pipeline Event System (Observer Pattern).
\"\"\"
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    \"\"\"Lifecycle events for the pipeline.\"\"\"
    BEFORE_PREDICTION = "before_prediction"
    AFTER_PREDICTION  = "after_prediction"
    BEFORE_STAGE      = "before_stage"
    AFTER_STAGE       = "after_stage"
    STAGE_SKIPPED     = "stage_skipped"
    STAGE_ERROR       = "stage_error"
    PIPELINE_ERROR    = "pipeline_error"
    PIPELINE_REJECTED = "pipeline_rejected"


@dataclass
class EventPayload:
    \"\"\"Context passed to event handlers.\"\"\"
    event_type: EventType
    request_id: str
    timestamp_utc: str
    stage_name: str | None = None
    error: Exception | None = None
    extra: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[EventPayload], None]


class EventDispatcher:
    \"\"\"
    Thread-safe fail-safe event dispatcher for observability.
    Runs on a per-executor basis.
    \"\"\"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # dict of event_type -> list of (priority, handler)
        self._handlers: dict[EventType, list[tuple[int, EventHandler]]] = {
            e: [] for e in EventType
        }

    def subscribe(self, event_type: EventType, handler: EventHandler, priority: int = 0) -> None:
        \"\"\"Add a listener for a specific event type.\"\"\"
        with self._lock:
            self._handlers[event_type].append((priority, handler))
            self._handlers[event_type].sort(key=lambda x: x[0], reverse=True)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        \"\"\"Remove a listener.\"\"\"
        with self._lock:
            self._handlers[event_type] = [
                (p, h) for p, h in self._handlers[event_type] if h != handler
            ]

    def emit(self, payload: EventPayload) -> None:
        \"\"\"
        Dispatch event to all subscribers sequentially.
        Guaranteed not to raise exceptions.
        \"\"\"
        with self._lock:
            handlers = list(self._handlers[payload.event_type])
        
        for priority, handler in handlers:
            try:
                handler(payload)
            except Exception as e:
                # Fail-safe boundary: never crash the pipeline
                logger.error(f"Event handler failed on {payload.event_type}: {e}")

    def clear(self) -> None:
        \"\"\"Remove all listeners.\"\"\"
        with self._lock:
            for e in EventType:
                self._handlers[e].clear()

    def handler_count(self, event_type: EventType) -> int:
        \"\"\"Return number of active listeners for an event type.\"\"\"
        with self._lock:
            return len(self._handlers[event_type])
"""

files["src/modules/inference_engine/device_manager.py"] = """
\"\"\"
Backend-agnostic device manager.
\"\"\"
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import AppSettings


class DeviceManager:
    \"\"\"Immutable runtime device information.\"\"\"
    
    def __init__(self, device_str: str) -> None:
        self._device_str = device_str
        self._is_cpu = device_str.lower() == "cpu"
        self._is_cuda = device_str.lower().startswith("cuda")
        self._is_mps = device_str.lower() == "mps"

    @property
    def device_str(self) -> str:
        return self._device_str

    @property
    def is_cuda(self) -> bool:
        return self._is_cuda

    @property
    def is_cpu(self) -> bool:
        return self._is_cpu

    @property
    def is_mps(self) -> bool:
        return self._is_mps

    @classmethod
    def auto_detect(cls, preferred: str = "cuda") -> "DeviceManager":
        \"\"\"Auto detect device dynamically without top-level torch import.\"\"\"
        if preferred.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    return cls(preferred)
            except ImportError:
                pass
        elif preferred == "mps":
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return cls("mps")
            except ImportError:
                pass
        return cls("cpu")

    @classmethod
    def from_config(cls, settings: AppSettings) -> "DeviceManager":
        \"\"\"Create from app configuration.\"\"\"
        return cls.auto_detect(settings.model.device)

    @classmethod
    def cpu(cls) -> "DeviceManager":
        \"\"\"Force CPU device.\"\"\"
        return cls("cpu")

    def __repr__(self) -> str:
        return f"DeviceManager(device='{self._device_str}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceManager):
            return False
        return self._device_str == other._device_str
"""

files["src/modules/inference_engine/factory.py"] = """
\"\"\"
Object Factory and Pipeline Builder.
\"\"\"
from __future__ import annotations

import importlib
from enum import Enum
from typing import Any, TypeVar

from src.framework.common.registry import ComponentRegistry
from src.modules.inference_engine.context import PipelineConfig
from src.modules.inference_engine.device_manager import DeviceManager
from src.modules.inference_engine.events import EventDispatcher

T = TypeVar("T")


class ObjectFactory:
    \"\"\"Configuration-driven dependency injection factory.\"\"\"

    @staticmethod
    def create_from_dict(config: dict[str, Any], registry: ComponentRegistry[Any] | None = None) -> Any:
        \"\"\"
        Dynamically instantiate an object from a config dictionary.
        Supports fetching from registry if 'type' matches a registry key,
        or dynamically importing if 'target' specifies a module.class path.
        \"\"\"
        if registry is not None and "type" in config:
            return registry.get(config["type"])

        if "target" in config:
            target_path = config["target"]
            module_path, class_name = target_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            params = config.get("params", {})
            return cls(**params)

        raise ValueError("Config must contain 'type' (for registry) or 'target' (for dynamic import).")


class StageSlot(str, Enum):
    \"\"\"Ordered slots for pipeline execution.\"\"\"
    QUALITY_ASSESSMENT = "quality_assessment"
    PREPROCESSING      = "preprocessing"
    CLASSIFICATION     = "classification"
    OOD_DETECTION      = "ood_detection"
    UNCERTAINTY        = "uncertainty_estimation"
    CALIBRATION        = "calibration"
    EXPLAINABILITY     = "explainability"
    RECOMMENDATION     = "recommendation"


STAGE_SLOT_ORDER: tuple[StageSlot, ...] = (
    StageSlot.QUALITY_ASSESSMENT,
    StageSlot.PREPROCESSING,
    StageSlot.CLASSIFICATION,
    StageSlot.OOD_DETECTION,
    StageSlot.UNCERTAINTY,
    StageSlot.CALIBRATION,
    StageSlot.EXPLAINABILITY,
    StageSlot.RECOMMENDATION,
)


class PipelineBuilder:
    \"\"\"Fluent builder for inference pipelines.\"\"\"

    def __init__(self) -> None:
        self._stages: dict[StageSlot, Any] = {}
        self._dispatcher: EventDispatcher | None = None
        self._device_manager: DeviceManager | None = None
        self._config: PipelineConfig | None = None
        self._frozen = False

    def _add_stage(self, slot: StageSlot, stage: Any) -> "PipelineBuilder":
        if self._frozen:
            raise RuntimeError("Cannot add stages to a frozen builder.")
        self._stages[slot] = stage
        return self

    def with_quality_assessor(self, stage: Any) -> "PipelineBuilder":
        return self._add_stage(StageSlot.QUALITY_ASSESSMENT, stage)

    def with_preprocessor(self, stage: Any) -> "PipelineBuilder":
        return self._add_stage(StageSlot.PREPROCESSING, stage)

    def with_classifier(self, stage: Any) -> "PipelineBuilder":
        return self._add_stage(StageSlot.CLASSIFICATION, stage)

    def with_ood_detector(self, stage: Any) -> "PipelineBuilder":
        return self._add_stage(StageSlot.OOD_DETECTION, stage)

    def with_uncertainty_estimator(self, stage: Any) -> "PipelineBuilder":
        return self._add_stage(StageSlot.UNCERTAINTY, stage)

    def with_calibrator(self, stage: Any) -> "PipelineBuilder":
        return self._add_stage(StageSlot.CALIBRATION, stage)

    def with_explainer(self, stage: Any) -> "PipelineBuilder":
        return self._add_stage(StageSlot.EXPLAINABILITY, stage)

    def with_recommendation_engine(self, stage: Any) -> "PipelineBuilder":
        return self._add_stage(StageSlot.RECOMMENDATION, stage)

    def with_event_dispatcher(self, dispatcher: EventDispatcher) -> "PipelineBuilder":
        if self._frozen:
            raise RuntimeError("Cannot modify a frozen builder.")
        self._dispatcher = dispatcher
        return self

    def with_device_manager(self, dm: DeviceManager) -> "PipelineBuilder":
        if self._frozen:
            raise RuntimeError("Cannot modify a frozen builder.")
        self._device_manager = dm
        return self

    def with_config(self, config: PipelineConfig) -> "PipelineBuilder":
        if self._frozen:
            raise RuntimeError("Cannot modify a frozen builder.")
        self._config = config
        return self

    def freeze(self) -> None:
        \"\"\"Freeze builder to prevent further modifications.\"\"\"
        self._frozen = True

    def build(self) -> Any:
        \"\"\"
        Produce immutable execution graph (PipelineExecutor).
        \"\"\"
        self.freeze()
        
        ordered_stages = []
        for slot in STAGE_SLOT_ORDER:
            if slot in self._stages:
                ordered_stages.append(self._stages[slot])

        # Validate core dependencies
        if StageSlot.CLASSIFICATION in self._stages and StageSlot.PREPROCESSING not in self._stages:
            raise ValueError("Classification stage requires a preprocessing stage.")
        if StageSlot.RECOMMENDATION in self._stages and StageSlot.CLASSIFICATION not in self._stages:
            raise ValueError("Recommendation stage requires a classification stage.")

        # In 5.2.2 we do not implement PipelineExecutor instantiation yet.
        return {
            "stages": ordered_stages,
            "dispatcher": self._dispatcher,
            "device_manager": self._device_manager,
            "config": self._config,
        }
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
