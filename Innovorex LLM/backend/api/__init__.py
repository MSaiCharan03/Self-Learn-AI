from .auth import router as auth_router
from .chat import router as chat_router
from .feedback import router as feedback_router
from .search import router as search_router
from .models import router as models_router
from .training import router as training_router
from .external import router as external_router

__all__ = [
    "auth_router",
    "chat_router", 
    "feedback_router",
    "search_router",
    "models_router",
    "training_router",
    "external_router"
]