"""
Model Loading and Caching.
"""
from __future__ import annotations

import threading
from typing import Any

from src.modules.inference_engine.device_manager import DeviceManager
from src.modules.model_registry.adapter import BackendAdapter
from src.modules.model_registry.descriptor import ModelDescriptor
from src.modules.model_registry.resolver import WeightResolverProtocol


class ModelLoader:
    """
    Handles lazy loading, integrity validation, and cache management
    using injected resolvers and backend adapters.
    """
    def __init__(
        self,
        resolver: WeightResolverProtocol,
        adapters: dict[str, BackendAdapter]
    ) -> None:
        self.resolver = resolver
        self.adapters = adapters
        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _get_cache_key(self, descriptor: ModelDescriptor, device: DeviceManager) -> str:
        return f"{descriptor.model_id}@{descriptor.version}:{device.device_str}"

    def load(self, descriptor: ModelDescriptor, device: DeviceManager) -> Any:
        cache_key = self._get_cache_key(descriptor, device)

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

            # 1. Resolve Path
            path = self.resolver.resolve(descriptor.model_id, descriptor.version)

            # 2. Checksum validation
            if descriptor.checksum:
                if not self.resolver.validate_checksum(path, descriptor.checksum):
                    raise ValueError(f"Checksum validation failed for model {descriptor.model_id}")

            # 3. Select backend adapter
            if descriptor.backend not in self.adapters:
                raise ValueError(f"Backend adapter '{descriptor.backend}' is not supported.")
            adapter = self.adapters[descriptor.backend]

            # 4. Load
            model = adapter.load_model(path, device)

            # 5. Cache
            self._cache[cache_key] = model
            return model

    def unload(self, descriptor: ModelDescriptor, device: DeviceManager) -> None:
        cache_key = self._get_cache_key(descriptor, device)
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
