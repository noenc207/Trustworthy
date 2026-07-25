"""
Dependency Injection Container.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

class DIContainer:
    """
    Dependency Injection Container.
    Supports singletons and transient resolution.
    No global state. Instance-based.
    """
    def __init__(self) -> None:
        self._singletons: dict[type, Any] = {}
        self._factories: dict[type, Callable[..., Any]] = {}

    def register_singleton(self, interface: type[T], instance: T) -> None:
        """Register a singleton instance for an interface."""
        self._singletons[interface] = instance

    def register_factory(self, interface: type[T], factory: Callable[..., T]) -> None:
        """Register a transient factory for an interface."""
        self._factories[interface] = factory

    def resolve(self, interface: type[T]) -> T:
        """Resolve a dependency by interface."""
        if interface in self._singletons:
            return self._singletons[interface]

        if interface in self._factories:
            return self._factories[interface]()

        raise KeyError(f"No DI registration found for {interface}")
