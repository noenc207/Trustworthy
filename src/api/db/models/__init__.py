"""
Database Models Package.
All models must be imported here so Alembic can discover them.
"""
from src.api.db.models.prediction import Prediction
from src.api.db.models.upload import Upload
from src.api.db.models.user import User

__all__ = [
    "Prediction",
    "Upload",
    "User",
]
