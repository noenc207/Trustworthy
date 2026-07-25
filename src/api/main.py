"""
FastAPI Application Entry Point.

Sets up:
  - CORS middleware
  - Exception handlers
  - Router registration
  - Lifespan (startup/shutdown)
  - Prometheus metrics
  - OpenAPI documentation
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

try:
    from prometheus_fastapi_instrumentator import Instrumentator as _Instrumentator
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _Instrumentator = None  # type: ignore[assignment,misc]
    _PROMETHEUS_AVAILABLE = False

from src.api.db import dispose_engine
from src.api.dependencies.redis import init_redis, close_redis
from src.api.middleware.logging import RequestLoggingMiddleware
from src.core.config import get_settings
from src.core.exceptions import SkinAIException
from src.core.logging import setup_logging
from src.api.routers import health, prediction, auth, upload

settings = get_settings()
setup_logging(settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.app_env}")

    # ── Startup ──────────────────────────────────────────────────────
    # The DB engine connection pool is created lazily on first query.
    logger.info("Database engine initialised (pool will connect on first use)")
    
    # Initialize Redis connection pool
    await init_redis()
    logger.info("Redis connection pool initialized")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Closing Redis connection pool…")
    await close_redis()
    
    logger.info("Draining database connection pool…")
    await dispose_engine()
    logger.info("Shutdown complete")


# ── Application Factory ──────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Trustworthy AI Platform for Skin Lesion Analysis — REST API",
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
        openapi_url="/openapi.json" if settings.app_env != "production" else None,
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────
    # Add Request Logging Middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ─────────────────────────────────────────
    @app.exception_handler(SkinAIException)
    async def skin_ai_exception_handler(
        request: Request, exc: SkinAIException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    # ── Routers ────────────────────────────────────────────────────
    # Routers are added incrementally per milestone.
    # Add a router here once its milestone is complete.
    api_prefix = "/api/v1"
    app.include_router(health.router, tags=["Health"])
    app.include_router(prediction.router, prefix=f"{api_prefix}/predict", tags=["Prediction"])
    app.include_router(auth.router, prefix=f"{api_prefix}/auth", tags=["Auth"])
    app.include_router(upload.router, prefix=f"{api_prefix}/upload", tags=["Upload"])
    # Milestone 7 ↓ history.router / gradcam.router / ood.router

    # ── Prometheus Metrics (optional — requires prometheus_fastapi_instrumentator) ──
    if _PROMETHEUS_AVAILABLE and _Instrumentator is not None:
        _Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
        ).instrument(app).expose(app, endpoint="/metrics")
    else:
        logger.warning(  # noqa: G004
            "prometheus_fastapi_instrumentator not installed — /metrics endpoint disabled"
        )

    return app


app = create_app()
