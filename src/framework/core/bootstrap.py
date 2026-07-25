"""
Framework Bootstrap Sequence.
"""
from __future__ import annotations

from pathlib import Path

from src.framework.core.config import ConfigurationManager, FrameworkConfig
from src.framework.core.di import DIContainer
from src.framework.core.health import HealthCheckSystem
from src.framework.core.logging import LoggingInfrastructure
from src.modules.inference_engine.device_manager import DeviceManager
from src.modules.inference_engine.events import EventDispatcher
from src.modules.model_registry.loader import ModelLoader
from src.modules.model_registry.registry import ModelRegistry
from src.modules.model_registry.resolver import LocalWeightResolver


class FrameworkBootstrap:
    """
    Deterministic startup sequence for the framework.
    Initializes DI, logging, configurations, and core systems.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.container = DIContainer()

    def boot(self) -> DIContainer:
        # 1. Config
        config = ConfigurationManager.load(self.config_path)
        self.container.register_singleton(FrameworkConfig, config)

        # 2. Logging
        LoggingInfrastructure.setup(config.log_level)

        # 3. Device Manager
        device_manager = DeviceManager.auto_detect(config.device)
        self.container.register_singleton(DeviceManager, device_manager)

        # 4. Event System
        event_dispatcher = EventDispatcher()
        self.container.register_singleton(EventDispatcher, event_dispatcher)

        # 5. Model Infrastructure
        resolver = LocalWeightResolver(Path(config.model_registry_path))
        loader = ModelLoader(resolver, adapters={})
        model_registry = ModelRegistry(loader)
        self.container.register_singleton(ModelRegistry, model_registry)

        # 6. Health
        health_system = HealthCheckSystem()
        health_system.register_component("config", config)
        health_system.register_component("device_manager", device_manager)
        health_system.register_component("event_dispatcher", event_dispatcher)
        health_system.register_component("model_registry", model_registry)
        self.container.register_singleton(HealthCheckSystem, health_system)

        return self.container
