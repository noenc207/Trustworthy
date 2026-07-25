"""
User Repository.
Handles all database operations for the User model.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models.user import User
from src.api.db.repositories.base import BaseRepository
from src.api.schemas.auth import UserCreate
from src.api.security.password import get_password_hash


class UserRepository(BaseRepository[User, UserCreate, Any]):
    """
    Repository for User operations.
    Overrides `create` to automatically hash the password.
    """
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> User | None:
        """Fetch a user by their email address."""
        stmt = select(self.model).where(self.model.email == email)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_username(self, username: str) -> User | None:
        """Fetch a user by their username."""
        stmt = select(self.model).where(self.model.username == username)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def create(self, *, obj_in: UserCreate | dict[str, Any]) -> User:
        """
        Create a new user. Automatically hashes the password.
        """
        create_data = obj_in.copy() if isinstance(obj_in, dict) else obj_in.model_dump()

        if "password" in create_data:
            hashed_password = get_password_hash(create_data.pop("password"))
            create_data["hashed_password"] = hashed_password

        return await super().create(obj_in=create_data)
