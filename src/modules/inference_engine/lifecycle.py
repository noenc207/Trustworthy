"""
Pipeline lifecycle hooks.
"""
from __future__ import annotations

from collections.abc import Callable

from src.modules.inference_engine.context import PipelineContext
from src.modules.inference_engine.stages import PipelineStage


class PipelineLifecycle:
    """Registry for runtime lifecycle hook callbacks."""

    def __init__(self) -> None:
        self.before_pipeline: list[Callable[[PipelineContext], None]] = []
        self.after_pipeline: list[Callable[[PipelineContext], None]] = []

        self.before_stage: list[Callable[[PipelineStage, PipelineContext], None]] = []
        self.after_stage: list[Callable[[PipelineStage, PipelineContext], None]] = []

        self.before_cleanup: list[Callable[[PipelineContext], None]] = []
        self.after_cleanup: list[Callable[[PipelineContext], None]] = []

        self.on_failure: list[Callable[[PipelineContext, Exception], None]] = []
        self.on_cancel: list[Callable[[PipelineContext], None]] = []
