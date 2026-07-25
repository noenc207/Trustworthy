"""Core package for Trustworthy Skin Cancer Detection.

This package contains the core components of the application including
configuration, exception hierarchy, interfaces, and data transfer objects.
"""

from .config import AppConfig, get_config
from .dtos import PredictionRequestDTO, PredictionResponseDTO
from .exceptions import (
    ConfigError,
    ExplainabilityError,
    ModelError,
    StorageError,
    TrustworthyError,
    ValidationError,
)
from .interfaces import ExplainabilityInterface, ModelInterface, StorageInterface

__all__ = [
    "AppConfig",
    "ConfigError",
    "ExplainabilityError",
    "ExplainabilityInterface",
    "ModelError",
    "ModelInterface",
    "PredictionRequestDTO",
    "PredictionResponseDTO",
    "StorageError",
    "StorageInterface",
    "TrustworthyError",
    "ValidationError",
    "get_config",
]
