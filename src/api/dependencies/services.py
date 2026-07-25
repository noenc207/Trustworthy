"""
Service Dependencies.
Provides Dependency Injection for Repositories and Services.
"""
from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.session import get_async_session
from src.api.db.repositories.user import UserRepository
from src.api.db.repositories.upload import UploadRepository
from src.api.db.repositories.prediction import PredictionRepository
from src.api.services.auth import AuthService
from src.api.services.upload import UploadService
from src.api.services.prediction import PredictionService
from src.api.dependencies.engine import get_inference_engine
from src.modules.inference_engine.engine import TrustworthyInferenceEngine


def get_user_repository(session: AsyncSession = Depends(get_async_session)) -> UserRepository:
    """Dependency: Yields a UserRepository."""
    return UserRepository(session)


def get_auth_service(user_repo: UserRepository = Depends(get_user_repository)) -> AuthService:
    """Dependency: Yields an AuthService."""
    return AuthService(user_repo)


def get_upload_repository(session: AsyncSession = Depends(get_async_session)) -> UploadRepository:
    """Dependency: Yields an UploadRepository."""
    return UploadRepository(session)


def get_upload_service(upload_repo: UploadRepository = Depends(get_upload_repository)) -> UploadService:
    """Dependency: Yields an UploadService."""
    return UploadService(upload_repo)


def get_prediction_repository(session: AsyncSession = Depends(get_async_session)) -> PredictionRepository:
    """Dependency: Yields a PredictionRepository."""
    return PredictionRepository(session)


def get_prediction_service(
    prediction_repo: PredictionRepository = Depends(get_prediction_repository),
    upload_repo: UploadRepository = Depends(get_upload_repository),
    engine: TrustworthyInferenceEngine = Depends(get_inference_engine)
) -> PredictionService:
    """Dependency: Yields a PredictionService."""
    return PredictionService(prediction_repo, upload_repo, engine)
