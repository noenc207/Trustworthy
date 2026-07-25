"""
Upload Repository.
Handles all database operations for the Upload model.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models.upload import Upload
from src.api.db.repositories.base import BaseRepository


class UploadCreate(BaseModel):
    """Internal schema for creating an upload record."""
    user_id: UUID
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int


class UploadRepository(BaseRepository[Upload, UploadCreate, Any]):
    """
    Repository for Upload operations.
    """
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Upload, session)

    async def get_by_user(self, user_id: UUID | str) -> Sequence[Upload]:
        """Fetch all uploads belonging to a specific user."""
        stmt = select(self.model).where(self.model.user_id == user_id).order_by(self.model.created_at.desc())
        result = await self.session.execute(stmt)
        return result.scalars().all()
