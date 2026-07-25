"""
Upload Service.
Handles business logic for receiving, validating, and saving image uploads securely.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import IO
from uuid import UUID, uuid4

from anyio import to_thread
from fastapi import UploadFile

from src.api.db.models.upload import Upload
from src.api.db.repositories.upload import UploadCreate, UploadRepository
from src.core.config import get_settings
from src.core.exceptions import InvalidImageError, ValidationError

_settings = get_settings()


def _save_file_to_disk(source_file: IO[bytes], destination: Path) -> int:
    """
    Synchronous helper to save a file stream to disk using a blocking call.
    Should be executed in a thread pool.
    Returns the size of the saved file in bytes.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(source_file, buffer)
    return destination.stat().st_size


class UploadService:
    """Service handling secure file uploads and storage."""

    def __init__(self, upload_repo: UploadRepository) -> None:
        self.upload_repo = upload_repo
        self.upload_dir = _settings.storage.upload_dir
        self.allowed_extensions = _settings.storage.allowed_extensions
        self.max_size = _settings.storage.max_file_size_bytes

    async def process_upload(self, file: UploadFile, user_id: UUID | str) -> Upload:
        """
        Validates, saves the file to disk, and creates a database record.
        """
        # Validate file presence
        if not file.filename:
            raise ValidationError(message="No filename provided")

        # Validate extension
        ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
        if ext not in self.allowed_extensions:
            raise InvalidImageError(
                message=f"Extension '{ext}' not allowed. Allowed: {self.allowed_extensions}"
            )

        # Generate a safe, unique filename to prevent path traversal/overwrite
        unique_filename = f"{uuid4().hex}.{ext}"
        destination_path = self.upload_dir / unique_filename

        # Save to disk securely without blocking the async event loop
        # FastAPI's UploadFile.file is a SpooledTemporaryFile (standard IO)
        size_bytes = await to_thread.run_sync(
            _save_file_to_disk, file.file, destination_path
        )

        # Validate file size
        if size_bytes > self.max_size:
            # Cleanup if too large
            destination_path.unlink(missing_ok=True)
            raise InvalidImageError(
                message=f"File exceeds maximum allowed size ({self.max_size / 1024 / 1024:.2f} MB)"
            )
            
        if size_bytes == 0:
            destination_path.unlink(missing_ok=True)
            raise InvalidImageError(message="Uploaded file is empty")

        # Create database record
        upload_in = UploadCreate(
            user_id=UUID(str(user_id)),
            filename=unique_filename,
            original_filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=size_bytes,
        )
        
        return await self.upload_repo.create(obj_in=upload_in)
