"""
Model Lifecycle Management.
"""
from __future__ import annotations

import threading
from enum import Enum


class ModelLifecycleState(str, Enum):
    REGISTERED = "registered"
    LOADED = "loaded"
    READY = "ready"
    RUNNING = "running"
    UNLOADED = "unloaded"
    ARCHIVED = "archived"

class ModelLifecycleError(Exception):
    """Raised on invalid lifecycle transitions."""
    pass

class ModelLifecycle:
    """Thread-safe finite state machine for model lifecycles."""

    def __init__(self) -> None:
        self._state = ModelLifecycleState.REGISTERED
        self._lock = threading.RLock()
        self._allowed_transitions: dict[ModelLifecycleState, set[ModelLifecycleState]] = {
            ModelLifecycleState.REGISTERED: {ModelLifecycleState.LOADED, ModelLifecycleState.ARCHIVED},
            ModelLifecycleState.LOADED: {ModelLifecycleState.READY, ModelLifecycleState.UNLOADED},
            ModelLifecycleState.READY: {ModelLifecycleState.RUNNING, ModelLifecycleState.UNLOADED},
            ModelLifecycleState.RUNNING: {ModelLifecycleState.READY, ModelLifecycleState.UNLOADED},
            ModelLifecycleState.UNLOADED: {ModelLifecycleState.LOADED, ModelLifecycleState.ARCHIVED},
            ModelLifecycleState.ARCHIVED: set(),
        }

    @property
    def current(self) -> ModelLifecycleState:
        with self._lock:
            return self._state

    def transition_to(self, new_state: ModelLifecycleState) -> None:
        with self._lock:
            if new_state not in self._allowed_transitions[self._state]:
                raise ModelLifecycleError(
                    f"Cannot transition model from {self._state.value} to {new_state.value}"
                )
            self._state = new_state
