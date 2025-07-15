"""
Admin utilities and decorators for the Self-Learning LLM Platform
"""
from functools import wraps
from fastapi import HTTPException, status, Depends
from .security import get_current_active_user
from ..models.user import User, UserRole

def admin_required(func):
    """Decorator to require admin privileges"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract current_user from kwargs or find it in args
        current_user = kwargs.get('current_user')
        if not current_user:
            # Look for User object in args
            for arg in args:
                if isinstance(arg, User):
                    current_user = arg
                    break
        
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        if not current_user.is_admin():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        return await func(*args, **kwargs)
    return wrapper

async def get_current_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current user and verify admin privileges"""
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

def is_admin_user(user: User) -> bool:
    """Check if user has admin privileges"""
    return user.role == UserRole.ADMIN

def get_user_role(user: User) -> str:
    """Get user role as string"""
    return user.role.value if user.role else "user"