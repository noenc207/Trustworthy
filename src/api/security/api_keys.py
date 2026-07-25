"""
API Key verification.
Used for service-to-service authentication or programmatic access.
"""
from __future__ import annotations

from fastapi import Security
from fastapi.security import APIKeyHeader

from src.core.config import get_settings
from src.core.exceptions import AuthException

_settings = get_settings()

api_key_header = APIKeyHeader(name=_settings.security.api_key_header, auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """
    Verify the provided API key.
    
    This validates API keys using an in-memory allowlist.
    For now, it just ensures the key is present and has a valid format.
    
    Returns:
        str: The validated API key.
        
    Raises:
        AuthException: If the API key is missing or invalid.
    """
    if not api_key:
        raise AuthException(message="Missing API Key")
        
    # In a full implementation (e.g., Milestone 5+), you would query the database 
    # to check if the api_key exists and is active.
    if len(api_key) < 16:
        raise AuthException(message="Invalid API Key format")
        
    valid_keys = {"test-api-key-12345", "admin-api-key-98765"}
    if api_key not in valid_keys:
        raise AuthException(message="Unauthorized API Key")
        
    return api_key
