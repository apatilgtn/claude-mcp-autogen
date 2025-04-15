"""
API dependencies for the FastAPI application.
This module provides dependencies for authentication and other common functionality.
"""

import time
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from loguru import logger
from pydantic import BaseModel

from src.core.config import settings


# Define token scheme - Point to the actual token endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    exp: Optional[int] = None


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    roles: list[str] = []


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


# Mock user database (for development purposes)
# In a real application, you would use a proper database
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
        "disabled": False,
        "roles": ["admin"]
    },
    "user": {
        "username": "user",
        "full_name": "Regular User",
        "email": "user@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
        "disabled": False,
        "roles": ["user"]
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches hash, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


async def get_user(username: str) -> Optional[UserInDB]:
    """
    Get a user by username.
    
    Args:
        username: Username
        
    Returns:
        User if found, None otherwise
    """
    if username in FAKE_USERS_DB:
        user_dict = FAKE_USERS_DB[username]
        return UserInDB(**user_dict)
    return None


async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Authenticate a user.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        User if authentication successful, None otherwise
    """
    user = await get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[int] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Token data
        expires_delta: Token expiration time in seconds
        
    Returns:
        JWT token
    """
    to_encode = data.copy()
    
    # Get secret key from settings
    secret_key = settings.get("secret_key")
    if not secret_key:
        logger.error("No SECRET_KEY set in environment or config")
        secret_key = "INSECURE_DEFAULT_KEY_DO_NOT_USE_IN_PRODUCTION"  # Fallback for development
    
    # Set expiration time
    expire = time.time() + (expires_delta or 86400)  # Default 24 hours
    to_encode.update({"exp": expire})
    
    # Create JWT token
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get the current authenticated user.
    
    Args:
        token: JWT token
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Get secret key from settings
        secret_key = settings.get("secret_key")
        if not secret_key:
            logger.error("No SECRET_KEY set in environment or config")
            secret_key = "INSECURE_DEFAULT_KEY_DO_NOT_USE_IN_PRODUCTION"  # Fallback for development
        
        # Decode JWT token
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        username: str = payload.get("sub")
        exp: int = payload.get("exp")
        
        if username is None:
            raise credentials_exception
        
        # Check token expiration
        if exp is None or exp < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token_data = TokenData(username=username, exp=exp)
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = await get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    # Return user without hashed password
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        roles=user.roles
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get the current active user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current active user
        
    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def verify_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Verify that the current user is an admin.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current admin user
        
    Raises:
        HTTPException: If user is not an admin
    """
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


# For development/testing - allows bypassing authentication
async def get_optional_user(request: Request):
    """
    Get the current user if authenticated, or None if not.
    For development purposes to allow certain routes without authentication.
    
    Args:
        request: HTTP request
        
    Returns:
        Current user or None
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.replace("Bearer ", "")
    try:
        user = await get_current_user(token)
        return user
    except HTTPException:
        return None