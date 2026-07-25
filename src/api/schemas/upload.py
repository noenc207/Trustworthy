"""
Upload API Schemas.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    """Response returned after successfully uploading an image."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    created_at: datetime
