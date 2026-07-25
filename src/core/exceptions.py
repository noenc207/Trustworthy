"""Custom exception hierarchy for the core application.

This module defines all custom exceptions used throughout the application.
"""


class TrustworthyError(Exception):
    """Base exception for all Trustworthy application errors.

    This serves as the root for all custom exceptions to allow catching
    any application-specific error easily.
    """

    def __init__(self, message: str) -> None:
        """Initialize the base exception.

        Args:
            message: The error message detailing the issue.
        """
        super().__init__(message)
        self.message = message


class ConfigError(TrustworthyError):
    """Raised when there is a configuration error."""


class ModelError(TrustworthyError):
    """Raised when the machine learning model encounters an error."""


class StorageError(TrustworthyError):
    """Raised when a storage operation fails."""


class ExplainabilityError(TrustworthyError):
    """Raised when generating explainability artifacts fails."""


class ValidationError(TrustworthyError):
    """Raised when input data validation fails."""
