"""
Main Model Registry.
"""
from __future__ import annotations

import threading
from typing import Any

from src.modules.inference_engine.device_manager import DeviceManager
from src.modules.model_registry.descriptor import ModelDescriptor
from src.modules.model_registry.lifecycle import ModelLifecycle, ModelLifecycleState
from src.modules.model_registry.loader import ModelLoader


class ModelRegistry:
    """
    Model Registry for managing model registration, versioning, 
    aliases, metadata, and lazy loading support.
    """
    def __init__(self, loader: ModelLoader) -> None:
        self._loader = loader
        self._descriptors: dict[str, ModelDescriptor] = {}
        self._aliases: dict[str, str] = {}
        self._lifecycles: dict[str, ModelLifecycle] = {}
        self._lock = threading.RLock()

    def register(self, descriptor: ModelDescriptor) -> None:
        """Register a new model version descriptor."""
        key = f"{descriptor.model_id}@{descriptor.version}"
        with self._lock:
            if key in self._descriptors:
                raise ValueError(f"Model {key} is already registered.")
            self._descriptors[key] = descriptor
            self._lifecycles[key] = ModelLifecycle()

    def set_alias(self, alias: str, model_id: str, version: str) -> None:
        """Set a user-friendly alias pointing to a specific model version."""
        key = f"{model_id}@{version}"
        with self._lock:
            if key not in self._descriptors:
                raise ValueError(f"Model {key} is not registered.")
            self._aliases[alias] = key

    def get_descriptor(self, identifier: str) -> ModelDescriptor:
        """Immutable lookup for a model descriptor by ID/version or alias."""
        with self._lock:
            key = self._aliases.get(identifier, identifier)
            if key not in self._descriptors:
                raise KeyError(f"Model '{identifier}' not found in registry.")
            return self._descriptors[key]

    def load_model(self, identifier: str, device: DeviceManager) -> Any:
        """Lazy load a model and transition its lifecycle."""
        descriptor = self.get_descriptor(identifier)
        key = f"{descriptor.model_id}@{descriptor.version}"

        with self._lock:
            lifecycle = self._lifecycles[key]

            if lifecycle.current in {ModelLifecycleState.REGISTERED, ModelLifecycleState.UNLOADED}:
                model = self._loader.load(descriptor, device)
                lifecycle.transition_to(ModelLifecycleState.LOADED)
                lifecycle.transition_to(ModelLifecycleState.READY)
                return model

            elif lifecycle.current in {ModelLifecycleState.READY, ModelLifecycleState.RUNNING}:
                return self._loader.load(descriptor, device)  # Fetch from cache safely

            else:
                raise ValueError(
                    f"Cannot load model '{identifier}' in state: {lifecycle.current.value}"
                )

    def unload_model(self, identifier: str, device: DeviceManager) -> None:
        """Unload a model from cache and transition its lifecycle."""
        descriptor = self.get_descriptor(identifier)
        key = f"{descriptor.model_id}@{descriptor.version}"

        with self._lock:
            lifecycle = self._lifecycles[key]
            self._loader.unload(descriptor, device)
            if lifecycle.current != ModelLifecycleState.UNLOADED:
                lifecycle.transition_to(ModelLifecycleState.UNLOADED)
