"""
Unified Framework Facade.
"""
from __future__ import annotations

from typing import Any

from src.framework.common.results import PipelineResult
from src.framework.core.bootstrap import FrameworkBootstrap
from src.framework.core.di import DIContainer
from src.framework.core.health import HealthCheckSystem, HealthStatus
from src.modules.inference_engine.context import PipelineConfig
from src.modules.inference_engine.executor import PipelineExecutor


class Framework:
    """
    Unified public API for the TrustDerm AI Framework.
    The application must never directly communicate with internal modules.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self._bootstrap = FrameworkBootstrap(config_path)
        self._container: DIContainer | None = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Execute deterministic startup sequence."""
        if self._is_initialized:
            return
        self._container = self._bootstrap.boot()
        self._is_initialized = True

    def health(self) -> HealthStatus:
        """Return structured health status of the framework."""
        if not self._is_initialized or self._container is None:
            return HealthStatus(status="DOWN", components={"framework": "not_initialized"})

        health_system = self._container.resolve(HealthCheckSystem)
        return health_system.health()

    def execute(self, executor: PipelineExecutor, raw_image: Any, config: PipelineConfig) -> PipelineResult:
        """Run an inference pipeline."""
        if not self._is_initialized:
            raise RuntimeError("Framework must be initialized before execution.")

        return executor.execute(raw_image, config)

    def shutdown(self) -> None:
        """Graceful shutdown sequence."""
        if not self._is_initialized:
            return

        # Clean up resources
        if self._container:
            try:
                from src.modules.inference_engine.events import EventDispatcher
                dispatcher = self._container.resolve(EventDispatcher)
                dispatcher.clear()
            except KeyError:
                import logging
                logging.getLogger(__name__).debug("EventDispatcher not resolved during shutdown.")

        self._container = None
        self._is_initialized = False
