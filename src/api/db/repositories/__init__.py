"""
Repositories Package.
Exports all repository classes for clean imports across the service layer.
"""
from src.api.db.repositories.base import BaseRepository
from src.api.db.repositories.prediction import PredictionCreate, PredictionRepository
from src.api.db.repositories.upload import UploadCreate, UploadRepository
from src.api.db.repositories.user import UserRepository

__all__ = [
    "BaseRepository",
    "PredictionCreate",
    "PredictionRepository",
    "UploadCreate",
    "UploadRepository",
    "UserRepository",
]
