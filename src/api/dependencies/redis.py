"""
Redis connection dependencies.
Provides connection pooling for caching and rate limiting.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from redis.asyncio import Redis, from_url

from src.core.config import get_settings

_settings = get_settings()

# Global connection pool
_redis_pool: Redis | None = None


async def init_redis() -> None:
    """Initialize the global Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = from_url(
            str(_settings.redis.url),
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5.0,
        )


async def close_redis() -> None:
    """Close the global Redis connection pool."""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.aclose()  # type: ignore[attr-defined]
        _redis_pool = None


async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    FastAPI dependency that yields the Redis connection pool.
    """
    if _redis_pool is None:
        await init_redis()
    yield _redis_pool  # type: ignore[misc]
