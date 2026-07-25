"""
Object Factory and Pipeline Builder.
"""
from __future__ import annotations

import importlib
from enum import StrEnum
from typing import Any, TypeVar

from src.framework.common.registry import ComponentRegistry
from src.modules.inference_engine.context import PipelineConfig
from src.modules.inference_engine.device_manager import DeviceManager
from src.modules.inference_engine.events import EventDispatcher

T = TypeVar("T")


class ObjectFactory:
    """Configuration-driven dependency injection factory."""

    @staticmethod
    def create_from_dict(config: dict[str, Any], registry: ComponentRegistry[Any] | None = None) -> Any:
        """
        Dynamically instantiate an object from a config dictionary.
        Supports fetching from registry if 'type' matches a registry key,
        or dynamically importing if 'target' specifies a module.class path.
        """
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


class StageSlot(StrEnum):
    """Ordered slots for pipeline execution."""
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
    """Fluent builder for inference pipelines."""

    def __init__(self) -> None:
        self._stages: dict[StageSlot, Any] = {}
        self._dispatcher: EventDispatcher | None = None
        self._device_manager: DeviceManager | None = None
        self._config: PipelineConfig | None = None
        self._frozen = False

    def _add_stage(self, slot: StageSlot, stage: Any) -> PipelineBuilder:
        if self._frozen:
            raise RuntimeError("Cannot add stages to a frozen builder.")
        self._stages[slot] = stage
        return self

    def with_quality_assessor(self, stage: Any) -> PipelineBuilder:
        return self._add_stage(StageSlot.QUALITY_ASSESSMENT, stage)

    def with_preprocessor(self, stage: Any) -> PipelineBuilder:
        return self._add_stage(StageSlot.PREPROCESSING, stage)

    def with_classifier(self, stage: Any) -> PipelineBuilder:
        return self._add_stage(StageSlot.CLASSIFICATION, stage)

    def with_ood_detector(self, stage: Any) -> PipelineBuilder:
        return self._add_stage(StageSlot.OOD_DETECTION, stage)

    def with_uncertainty_estimator(self, stage: Any) -> PipelineBuilder:
        return self._add_stage(StageSlot.UNCERTAINTY, stage)

    def with_calibrator(self, stage: Any) -> PipelineBuilder:
        return self._add_stage(StageSlot.CALIBRATION, stage)

    def with_explainer(self, stage: Any) -> PipelineBuilder:
        return self._add_stage(StageSlot.EXPLAINABILITY, stage)

    def with_recommendation_engine(self, stage: Any) -> PipelineBuilder:
        return self._add_stage(StageSlot.RECOMMENDATION, stage)

    def with_event_dispatcher(self, dispatcher: EventDispatcher) -> PipelineBuilder:
        if self._frozen:
            raise RuntimeError("Cannot modify a frozen builder.")
        self._dispatcher = dispatcher
        return self

    def with_device_manager(self, dm: DeviceManager) -> PipelineBuilder:
        if self._frozen:
            raise RuntimeError("Cannot modify a frozen builder.")
        self._device_manager = dm
        return self

    def with_config(self, config: PipelineConfig) -> PipelineBuilder:
        if self._frozen:
            raise RuntimeError("Cannot modify a frozen builder.")
        self._config = config
        return self

    def freeze(self) -> None:
        """Freeze builder to prevent further modifications."""
        self._frozen = True

    def build(self) -> Any:
        """
        Produce immutable execution graph (PipelineExecutor).
        """
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
