"""
Authentication Service.
Encapsulates business logic for user registration and login.
"""
from __future__ import annotations

from src.api.db.models.user import User
from src.api.db.repositories.user import UserRepository
from src.api.schemas.auth import Token, UserCreate
from src.api.security.jwt import create_access_token, create_refresh_token
from src.api.security.password import verify_password
from src.core.config import get_settings
from src.core.exceptions import AuthException, ValidationError

_settings = get_settings()


class AuthService:
    """Service handling authentication workflows."""

    def __init__(self, user_repo: UserRepository) -> None:
        self.user_repo = user_repo

    async def authenticate_user(self, username: str, password: str) -> User | None:
        """
        Authenticate a user by username and password.
        Returns the User object if successful, None otherwise.
        """
        user = await self.user_repo.get_by_username(username)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def register_user(self, user_in: UserCreate) -> User:
        """
        Register a new user.
        Validates that email and username are unique.
        """
        if await self.user_repo.get_by_email(user_in.email):
            raise ValidationError(message="Email already registered")
        
        if await self.user_repo.get_by_username(user_in.username):
            raise ValidationError(message="Username already taken")

        user = await self.user_repo.create(obj_in=user_in)
        return user

    def generate_tokens(self, user: User) -> Token:
        """
        Generate access and refresh tokens for a user.
        """
        access_token = create_access_token(
            subject=user.id,
            email=user.email,
            scopes=["admin"] if user.is_admin else ["user"],
        )
        refresh_token = create_refresh_token(subject=user.id)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=_settings.security.access_token_expire_minutes * 60,
        )
