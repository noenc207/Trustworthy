"""
Prediction Repository.
Handles all database operations for the Prediction model.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models.prediction import Prediction
from src.api.db.repositories.base import BaseRepository


class PredictionCreate(BaseModel):
    """Internal schema for creating a prediction record."""
    user_id: UUID
    upload_id: UUID
    predicted_class: str
    calibrated_confidence: float
    is_ood: bool
    full_result: dict[str, Any]


class PredictionRepository(BaseRepository[Prediction, PredictionCreate, Any]):
    """
    Repository for Prediction operations.
    """
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Prediction, session)

    async def get_by_upload(self, upload_id: UUID | str) -> Prediction | None:
        """Fetch the prediction associated with a specific upload."""
        stmt = select(self.model).where(self.model.upload_id == upload_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_user(self, user_id: UUID | str) -> Sequence[Prediction]:
        """Fetch all predictions belonging to a specific user."""
        stmt = select(self.model).where(self.model.user_id == user_id).order_by(self.model.created_at.desc())
        result = await self.session.execute(stmt)
        return result.scalars().all()
