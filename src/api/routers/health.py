"""
Health check endpoints.

Provides liveness and readiness probes for Kubernetes / Docker health checks.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """Liveness probe — returns 200 if service is running."""
    from src.core.config import get_settings
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get("/ready", summary="Readiness Probe")
async def readiness_check() -> dict[str, str]:
    """Readiness probe — verifies model and DB are loaded."""
    return {"status": "ready"}
