"""
Upload ORM Model.
Tracks files uploaded to the platform.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.db.base_model import AuditMixin, Base

if TYPE_CHECKING:
    from src.api.db.models.prediction import Prediction
    from src.api.db.models.user import User


class Upload(Base, AuditMixin):
    """
    Uploaded image metadata.
    Actual files are stored on disk (configured via StorageSettings),
    this table maps the disk file to the user and tracks metadata.
    """
    __tablename__ = "uploads"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="uploads")
    prediction: Mapped[Prediction] = relationship(
        "Prediction", back_populates="upload", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Upload {self.filename}>"
