"""
Pipeline Event System (Observer Pattern).
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    """Lifecycle events for the pipeline."""
    BEFORE_PREDICTION = "before_prediction"
    AFTER_PREDICTION  = "after_prediction"
    BEFORE_STAGE      = "before_stage"
    AFTER_STAGE       = "after_stage"
    STAGE_SKIPPED     = "stage_skipped"
    STAGE_ERROR       = "stage_error"
    PIPELINE_ERROR    = "pipeline_error"
    PIPELINE_REJECTED = "pipeline_rejected"


@dataclass
class EventPayload:
    """Context passed to event handlers."""
    event_type: EventType
    request_id: str
    timestamp_utc: str
    stage_name: str | None = None
    error: Exception | None = None
    extra: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[EventPayload], None]


class EventDispatcher:
    """
    Thread-safe fail-safe event dispatcher for observability.
    Runs on a per-executor basis.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # dict of event_type -> list of (priority, handler)
        self._handlers: dict[EventType, list[tuple[int, EventHandler]]] = {
            e: [] for e in EventType
        }

    def subscribe(self, event_type: EventType, handler: EventHandler, priority: int = 0) -> None:
        """Add a listener for a specific event type."""
        with self._lock:
            self._handlers[event_type].append((priority, handler))
            self._handlers[event_type].sort(key=lambda x: x[0], reverse=True)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a listener."""
        with self._lock:
            self._handlers[event_type] = [
                (p, h) for p, h in self._handlers[event_type] if h != handler
            ]

    def emit(self, payload: EventPayload) -> None:
        """
        Dispatch event to all subscribers sequentially.
        Guaranteed not to raise exceptions.
        """
        with self._lock:
            handlers = list(self._handlers[payload.event_type])

        for _priority, handler in handlers:
            try:
                handler(payload)
            except Exception as e:
                # Fail-safe boundary: never crash the pipeline
                logger.error(f"Event handler failed on {payload.event_type}: {e}")

    def clear(self) -> None:
        """Remove all listeners."""
        with self._lock:
            for e in EventType:
                self._handlers[e].clear()

    def handler_count(self, event_type: EventType) -> int:
        """Return number of active listeners for an event type."""
        with self._lock:
            return len(self._handlers[event_type])
