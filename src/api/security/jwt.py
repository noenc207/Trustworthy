"""
JWT creation and verification utilities.
Uses python-jose for RS256 / HS256 signature generation and validation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from jose import JWTError, jwt

from src.api.schemas.auth import TokenData
from src.core.config import get_settings
from src.core.exceptions import AuthException

_settings = get_settings()


def create_access_token(
    subject: str | UUID, email: str | None = None, scopes: list[str] | None = None
) -> str:
    """
    Create a short-lived JWT access token.
    """
    expire = datetime.utcnow() + timedelta(
        minutes=_settings.security.access_token_expire_minutes
    )
    to_encode: dict[str, Any] = {
        "exp": expire,
        "sub": str(subject),
        "type": "access",
    }
    if email:
        to_encode["email"] = email
    if scopes:
        to_encode["scopes"] = scopes

    return jwt.encode(
        to_encode,
        _settings.security.secret_key.get_secret_value(),
        algorithm=_settings.security.jwt_algorithm,
    )


def create_refresh_token(subject: str | UUID) -> str:
    """
    Create a long-lived JWT refresh token.
    """
    expire = datetime.utcnow() + timedelta(
        days=_settings.security.refresh_token_expire_days
    )
    to_encode: dict[str, Any] = {
        "exp": expire,
        "sub": str(subject),
        "type": "refresh",
    }
    return jwt.encode(
        to_encode,
        _settings.security.secret_key.get_secret_value(),
        algorithm=_settings.security.jwt_algorithm,
    )


def verify_token(token: str, expected_type: str = "access") -> TokenData:
    """
    Verify a JWT token signature and expiration.
    
    Returns:
        TokenData containing the extracted payload.
        
    Raises:
        AuthException: If the token is expired, invalid, or of the wrong type.
    """
    try:
        payload = jwt.decode(
            token,
            _settings.security.secret_key.get_secret_value(),
            algorithms=[_settings.security.jwt_algorithm],
        )
        
        token_type: str = payload.get("type", "")
        if token_type != expected_type:
            raise AuthException(message=f"Invalid token type: expected {expected_type}")

        subject: str | None = payload.get("sub")
        if subject is None:
            raise AuthException(message="Token missing subject claim")

        return TokenData(
            sub=subject,
            email=payload.get("email"),
            scopes=payload.get("scopes", []),
        )

    except JWTError as e:
        raise AuthException(message="Could not validate credentials", detail=str(e)) from e
