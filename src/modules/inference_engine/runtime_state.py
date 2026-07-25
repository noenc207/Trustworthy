"""
Pipeline Runtime State Management.
"""
from __future__ import annotations

import threading
from enum import StrEnum


class RuntimeState(StrEnum):
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateTransitionError(Exception):
    """Raised when an invalid runtime state transition is attempted."""


class RuntimeStateManager:
    """Thread-safe runtime state manager with transition validation."""

    def __init__(self) -> None:
        self._state = RuntimeState.CREATED
        self._lock = threading.RLock()
        self._allowed_transitions: dict[RuntimeState, set[RuntimeState]] = {
            RuntimeState.CREATED: {RuntimeState.INITIALIZING, RuntimeState.CANCELLED},
            RuntimeState.INITIALIZING: {RuntimeState.READY, RuntimeState.FAILED, RuntimeState.CANCELLED},
            RuntimeState.READY: {RuntimeState.RUNNING, RuntimeState.CANCELLED},
            RuntimeState.RUNNING: {RuntimeState.COMPLETED, RuntimeState.FAILED, RuntimeState.CANCELLED},
            RuntimeState.COMPLETED: set(),
            RuntimeState.FAILED: set(),
            RuntimeState.CANCELLED: set(),
        }

    @property
    def current(self) -> RuntimeState:
        with self._lock:
            return self._state

    def transition_to(self, new_state: RuntimeState) -> None:
        with self._lock:
            if new_state not in self._allowed_transitions[self._state]:
                raise StateTransitionError(
                    f"Invalid transition from {self._state.value} to {new_state.value}"
                )
            self._state = new_state
