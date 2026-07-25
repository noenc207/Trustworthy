"""
Framework Health Check System.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HealthStatus:
    status: str
    components: dict[str, str]

class HealthCheckSystem:
    """Centralized health check for framework components."""

    def __init__(self) -> None:
        self._checks: dict[str, Any] = {}

    def register_component(self, name: str, component: Any) -> None:
        self._checks[name] = component

    def health(self) -> HealthStatus:
        components_status = {}
        overall_status = "UP"

        for name, comp in self._checks.items():
            try:
                # Basic liveness probe
                if hasattr(comp, "health"):
                    comp_status = comp.health()
                else:
                    comp_status = "UP" if comp is not None else "DOWN"

                components_status[name] = comp_status
                if comp_status != "UP":
                    overall_status = "DEGRADED"
            except Exception:
                components_status[name] = "DOWN"
                overall_status = "DOWN"

        return HealthStatus(status=overall_status, components=components_status)
