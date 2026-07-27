"""
Authentication schemas.
"""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """JWT token payload."""
    sub: str | None = None  # Username or user ID
    email: str | None = None
    scopes: list[str] = Field(default_factory=list)


class UserResponse(BaseModel):
    """Public user info (never exposes password)."""
    id: UUID
    username: str
    email: str
    is_active: bool
    is_admin: bool
