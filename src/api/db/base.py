"""
Database engine and session factory.

Uses SQLAlchemy 2.0 async API with asyncpg driver.
The engine is a module-level singleton created once on import.
Session lifecycle is managed via get_async_session() dependency.

Design:
  - create_async_engine()  → single engine shared across all requests
  - async_sessionmaker()   → session factory (expire_on_commit=False avoids
                             lazy-load errors after commit in async context)
  - dispose_engine()       → called on app shutdown to drain connection pool
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import get_settings

_settings = get_settings()

# ---------------------------------------------------------------------------
# Engine — created once at module import time
# ---------------------------------------------------------------------------
engine: AsyncEngine = create_async_engine(
    str(_settings.db.url),          # PostgresDsn → str for SQLAlchemy
    pool_size=_settings.db.pool_size,
    max_overflow=_settings.db.max_overflow,
    pool_pre_ping=_settings.db.pool_pre_ping,
    echo=_settings.db.echo,
    # asyncpg-specific: prevents "connection already closed" under high load
    pool_recycle=1800,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
# expire_on_commit=False:
#   In async code, accessing attributes after commit would trigger a lazy load,
#   which is illegal in async SQLAlchemy. Setting this to False keeps the
#   in-memory state valid after commit without an extra DB round-trip.
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def dispose_engine() -> None:
    """
    Dispose the engine connection pool.
    Call this from the FastAPI lifespan shutdown hook.
    """
    await engine.dispose()
