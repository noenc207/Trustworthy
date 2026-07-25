"""
Authentication dependencies.
Uses OAuth2PasswordBearer to extract the JWT token from the Authorization header.
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

from src.api.schemas.auth import TokenData
from src.api.security.jwt import verify_token
from src.core.exceptions import AuthException

# The tokenUrl points to the login endpoint (to be implemented in Milestone 5)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_token_payload(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Extracts and verifies the JWT token from the request header.
    
    Returns:
        TokenData: The verified payload data.
        
    Raises:
        AuthException: If the token is missing, expired, or invalid.
    """
    if not token:
        raise AuthException(message="Not authenticated")
    return verify_token(token, expected_type="access")


async def get_current_user(
    token_data: TokenData = Depends(get_current_token_payload)
) -> dict[str, Any]:
    """
    FastAPI dependency: Returns the authenticated user payload.
    
    Currently returns a dict to satisfy existing routers (e.g., prediction.py).
    In Milestone 4/5 (Repositories/Services), this will be updated to query 
    the database and return a full User ORM object.
    
    Returns:
        dict: The user's token payload (containing 'sub', 'email', 'scopes').
    """
    return {
        "sub": token_data.sub,
        "email": token_data.email,
        "scopes": token_data.scopes,
    }
