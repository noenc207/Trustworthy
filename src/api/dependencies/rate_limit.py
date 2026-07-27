"""
Rate limiting dependency.
Uses Redis to enforce API quotas per user.
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends
from redis.asyncio import Redis

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.redis import get_redis
from src.core.config import get_settings
from src.core.exceptions import RateLimitError

_settings = get_settings()


class RateLimiter:
    """
    FastAPI dependency class for rate limiting.
    Tracks requests per user per minute using Redis sliding windows or simple counters.
    """
    def __init__(self, max_requests: int | None = None) -> None:
        self.max_requests = max_requests or _settings.security.rate_limit_per_minute

    async def __call__(
        self,
        current_user: dict[str, Any] = Depends(get_current_user),
        redis: Redis = Depends(get_redis),
    ) -> None:
        """
        Execute rate limit check.
        Uses a simple Redis string with TTL for the current minute window.
        """
        user_id = current_user["sub"]

        # Simple fixed-window rate limiting
        # Key rotates every minute
        import time
        current_minute = int(time.time() / 60)
        key = f"rate_limit:{user_id}:{current_minute}"

        # Increment counter
        count = await redis.incr(key)

        if count == 1:
            # First request in this minute, set TTL
            await redis.expire(key, 60)

        if count > self.max_requests:
            raise RateLimitError(
                message=f"Rate limit exceeded. Maximum {self.max_requests} requests per minute."
            )
