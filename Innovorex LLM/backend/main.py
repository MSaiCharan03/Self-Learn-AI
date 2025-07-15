from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import time
from contextlib import asynccontextmanager

from .api import auth_router, chat_router, feedback_router, search_router, models_router, training_router, external_router
from .core.database import create_tables
from .core.llm_manager import phi2_manager
from .utils.config import settings

# Create data directories
os.makedirs("./data/models", exist_ok=True)
os.makedirs("./data/faiss_index", exist_ok=True)
os.makedirs("./data/logs", exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("Starting LLM Platform...")
    create_tables()
    print("Database tables created/verified")
    
    # Initialize Phi-2 model
    try:
        print("Initializing Phi-2 model...")
        model_info = phi2_manager.get_model_info()
        if model_info["is_loaded"]:
            print(f"✅ Phi-2 model loaded successfully")
            print(f"   Model: {model_info['model_info']['model_name']}")
            print(f"   Vocab Size: {model_info['model_info']['vocab_size']}")
            print(f"   Context Length: {model_info['model_info']['context_length']}")
        else:
            print("❌ Phi-2 model failed to load")
    except Exception as e:
        print(f"❌ Error initializing Phi-2: {e}")
    
    # Initialize Faiss index and embedding model
    try:
        from .core.vector_store import vector_store
        from .core.embeddings import embedding_manager
        
        print("Initializing vector store and embeddings...")
        vector_stats = vector_store.get_stats()
        embedding_info = embedding_manager.get_model_info()
        
        print(f"✅ Vector store initialized with {vector_stats['total_embeddings']} embeddings")
        print(f"✅ Embedding model loaded: {embedding_info['model_name']}")
        
    except Exception as e:
        print(f"❌ Error initializing vector components: {e}")
    
    # Initialize external models
    try:
        from .core.external_models import external_model_manager
        
        print("Initializing external models...")
        available_models = external_model_manager.get_available_models()
        
        if available_models:
            print(f"✅ External models available: {', '.join(available_models)}")
        else:
            print("ℹ️  No external models configured (API keys not provided)")
            
    except Exception as e:
        print(f"❌ Error initializing external models: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down LLM Platform...")

# Create FastAPI app
app = FastAPI(
    title="Self-Learning LLM Platform",
    description="A scalable, modular, and self-improving LLM platform with local Phi-2, Faiss, and PostgreSQL",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"]
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Include API routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(training_router, prefix="/api/v1")
app.include_router(external_router, prefix="/api/v1")

# Serve static files (frontend)
if os.path.exists("../frontend"):
    app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/api/v1/")
async def root():
    return {
        "message": "Self-Learning LLM Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )