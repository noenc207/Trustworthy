"""
Thread-safe generic Component Registry.
Replaces all domain-specific registries.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar, Generic

T = TypeVar("T")


class RegistryFrozenError(Exception):
    """Raised when attempting to modify a frozen registry."""


class ComponentRegistry(Generic[T]):
    """
    Generic, thread-safe, typed component registry.
    Supports lazy instantiation and plugin discovery.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, T | Callable[[], T]] = {}
        self._is_factory: dict[str, bool] = {}
        self._lock = threading.RLock()
        self._frozen = False

    def freeze(self) -> None:
        """Freeze the registry to prevent further modifications."""
        with self._lock:
            self._frozen = True

    def register(self, key: str, value: T, overwrite: bool = False) -> None:
        """Register a concrete instance."""
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(f"Registry '{self.name}' is frozen.")
            if not overwrite and key in self._store:
                raise KeyError(f"Key '{key}' already registered in {self.name}.")
            self._store[key] = value
            self._is_factory[key] = False

    def register_factory(self, key: str, factory: Callable[[], T], overwrite: bool = False) -> None:
        """Register a factory for lazy instantiation."""
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(f"Registry '{self.name}' is frozen.")
            if not overwrite and key in self._store:
                raise KeyError(f"Key '{key}' already registered in {self.name}.")
            self._store[key] = factory
            self._is_factory[key] = True

    def unregister(self, key: str) -> None:
        """Remove a component from the registry."""
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(f"Registry '{self.name}' is frozen.")
            if key in self._store:
                del self._store[key]
                del self._is_factory[key]

    def contains(self, key: str) -> bool:
        """Check if a key exists in the registry."""
        with self._lock:
            return key in self._store

    def get(self, key: str) -> T:
        """Retrieve a component, instantiating it if registered as a factory."""
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
        """Return all registered keys."""
        with self._lock:
            return list(self._store.keys())

    def clear(self) -> None:
        """Clear all registrations. Used primarily for testing."""
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(f"Registry '{self.name}' is frozen.")
            self._store.clear()
            self._is_factory.clear()
