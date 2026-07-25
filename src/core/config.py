"""Configuration management for the core application.

This module provides the main application configuration using Pydantic settings.
It loads configuration from environment variables and provides defaults.
"""

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Fallback if pydantic_settings is not installed, though it should be in production
    from pydantic import BaseSettings, SettingsConfigDict  # type: ignore


class AppConfig(BaseSettings):
    """Main application configuration settings.

    This class manages all configuration required by the application.
    It reads from environment variables by default.

    Attributes:
        app_name: The name of the application.
        debug: Whether the application is running in debug mode.
        database_url: The connection string for the database.
        secret_key: Secret key for cryptographic operations.
        model_path: Path to the machine learning model.
    """

    app_name: str = "Trustworthy Skin Cancer Detection"
    debug: bool = False
    database_url: str = "sqlite:///./trustworthy.db"
    secret_key: str = "super-secret-key-for-dev-only"
    model_path: str = "./models/default_model.pt"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

def get_config() -> AppConfig:
    """Retrieves the application configuration.

    Returns:
        AppConfig: The current configuration instance.
    """
    return AppConfig()
