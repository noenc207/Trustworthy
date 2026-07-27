"""
Upload API Router.
Exposes endpoints for securely uploading skin lesion images.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile, status
from loguru import logger

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.services import get_upload_service
from src.api.schemas.upload import UploadResponse
from src.api.services.upload import UploadService

router = APIRouter()


@router.post(
    "/",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a skin lesion image",
    description="Securely uploads an image, validates extension/size, and prepares it for inference.",
)
async def upload_image(
    file: UploadFile = File(..., description="The image file to upload"),
    upload_service: UploadService = Depends(get_upload_service),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> Any:
    """Process and store an uploaded image."""
    user_id = current_user["sub"]
    logger.info(f"Processing upload for user_id={user_id}, filename={file.filename}")

    upload_record = await upload_service.process_upload(file=file, user_id=user_id)

    logger.info(f"Successfully saved upload_id={upload_record.id}")
    return upload_record
