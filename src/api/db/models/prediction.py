"""
Prediction ORM Model.
Stores inference results for analysis and historical lookup.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.db.base_model import AuditMixin, Base

if TYPE_CHECKING:
    from src.api.db.models.upload import Upload
    from src.api.db.models.user import User


class Prediction(Base, AuditMixin):
    """
    AI Prediction result.
    Key metrics (confidence, OOD status, predicted class) are extracted
    as top-level columns for fast filtering and analytics.
    The complete detailed result (heatmaps, full probability array)
    is stored in the full_result JSONB column.
    """
    __tablename__ = "predictions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    upload_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Fast filtering columns
    predicted_class: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    calibrated_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_ood: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)

    # Full prediction payload
    full_result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="predictions")
    upload: Mapped[Upload] = relationship("Upload", back_populates="prediction")

    def __repr__(self) -> str:
        return f"<Prediction {self.predicted_class} ({self.calibrated_confidence:.2f})>"
