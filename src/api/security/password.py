"""
Password hashing and verification using passlib and bcrypt.
"""
from passlib.context import CryptContext

# Create a single CryptContext instance for the application
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against its bcrypt hash.
    
    Args:
        plain_password: The plaintext password from user input.
        hashed_password: The bcrypt hash from the database.
        
    Returns:
        bool: True if the password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Generate a bcrypt hash for a plaintext password.
    
    Args:
        password: The plaintext password.
        
    Returns:
        str: The bcrypt hash string.
    """
    return pwd_context.hash(password)
