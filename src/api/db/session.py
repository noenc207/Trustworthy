"""
Async database session dependency for FastAPI.

Provides get_async_session() as a FastAPI dependency that yields
an AsyncSession per request and guarantees proper cleanup.

Usage in router:
    @router.get("/items")
    async def list_items(
        db: AsyncSession = Depends(get_async_session),
    ) -> list[Item]:
        ...

Transaction contract:
  - The session is committed only if the route handler returns normally.
  - On any exception the session is rolled back.
  - The session is always closed in the finally block.
  - Repositories call db.flush() (not commit) to stage changes within a
    request. The dependency commits at the boundary — keeping transaction
    control at the HTTP layer, not inside repositories.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.base import AsyncSessionLocal


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: yield an AsyncSession for the current request.

    The outer try/except/finally ensures:
      - commit() on clean exit
      - rollback() on any exception (preserves DB integrity)
      - close() always (returns connection to pool)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
