"""
Authentication API Router.
Exposes endpoints for user registration and login (token generation).
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger

from src.api.dependencies.services import get_auth_service
from src.api.schemas.auth import Token, UserCreate, UserResponse
from src.api.services.auth import AuthService
from src.core.exceptions import AuthException

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Creates a new user account with hashed password.",
)
async def register(
    user_in: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
) -> Any:
    """Register a new user."""
    logger.info(f"Registering new user: {user_in.username}")
    user = await auth_service.register_user(user_in)
    return user


@router.post(
    "/login",
    response_model=Token,
    summary="Login and get access token",
    description="Authenticates using OAuth2 password flow and returns JWT tokens.",
)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    auth_service: AuthService = Depends(get_auth_service),
) -> Token:
    """OAuth2 compatible token login, getting an access token for future requests."""
    user = await auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise AuthException(message="Incorrect username or password")
        
    if not user.is_active:
        logger.warning(f"Inactive user login attempt: {form_data.username}")
        raise AuthException(message="Inactive user account")
        
    logger.info(f"User logged in successfully: {user.username}")
    return auth_service.generate_tokens(user)
