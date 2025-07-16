#!/bin/bash

# =============================================================================
# Self-Learning LLM Platform - Complete Code Generator and Setup Script
# =============================================================================
# This script creates the entire Self-Learning LLM platform code structure
# from scratch, including all Python files, configuration, and setup.
# =============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

log_create() {
    echo -e "${CYAN}[CREATE]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/self-learning-llm"
VENV_PATH="$SCRIPT_DIR/.venv"
DEFAULT_HOST="localhost"
DEFAULT_PORT="8000"

# Parse command line arguments
HOST=${1:-$DEFAULT_HOST}
PORT=${2:-$DEFAULT_PORT}
SKIP_INSTALL=${3:-false}

# =============================================================================
# Directory Structure Creation
# =============================================================================

create_directory_structure() {
    log_step "Creating directory structure..."
    
    # Remove existing directory if it exists
    if [ -d "$PROJECT_ROOT" ]; then
        log_warning "Removing existing project directory..."
        rm -rf "$PROJECT_ROOT"
    fi
    
    # Create main directories
    DIRECTORIES=(
        "$PROJECT_ROOT"
        "$PROJECT_ROOT/backend"
        "$PROJECT_ROOT/backend/app"
        "$PROJECT_ROOT/backend/app/api"
        "$PROJECT_ROOT/backend/app/core"
        "$PROJECT_ROOT/backend/app/models"
        "$PROJECT_ROOT/backend/app/services"
        "$PROJECT_ROOT/backend/app/utils"
        "$PROJECT_ROOT/frontend"
        "$PROJECT_ROOT/scripts"
        "$PROJECT_ROOT/config"
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/models"
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/tests"
        "$PROJECT_ROOT/docs"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        mkdir -p "$dir"
        log_create "Created directory: ${dir#$SCRIPT_DIR/}"
    done
    
    log_success "Directory structure created"
}

# =============================================================================
# Requirements and Configuration Files
# =============================================================================

create_requirements_txt() {
    log_create "Creating requirements.txt..."
    
    cat > "$PROJECT_ROOT/requirements.txt" << 'EOF'
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.13.0

# Machine Learning
torch==2.1.1
transformers==4.36.0
sentence-transformers==2.7.0
faiss-cpu==1.7.4
numpy==1.24.3
scikit-learn==1.3.2
datasets==4.0.0

# HTTP and Security
httpx==0.25.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Configuration and Environment
python-dotenv==1.0.0
pyyaml==6.0.1

# SSH and Remote Access
paramiko==3.3.1
fabric==3.2.2

# Utilities
pandas==2.1.4
tqdm==4.67.1
asyncio==3.4.3
aiofiles==23.2.1

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
EOF
}

create_env_file() {
    log_create "Creating .env configuration..."
    
    cat > "$PROJECT_ROOT/config/.env" << EOF
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=self_learning_llm_dev
DB_USER=postgres
DB_PASSWORD=password
DATABASE_URL=postgresql://postgres:password@localhost:5432/self_learning_llm_dev

# Server Configuration
SERVER_HOST=$HOST
SERVER_PORT=$PORT
DEBUG=true

# Model Configuration
MODEL_CACHE_DIR=./models
PHI2_MODEL_PATH=./models/phi2
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Faiss Configuration
FAISS_INDEX_PATH=./data/faiss_index
VECTOR_DIMENSION=384

# Security (development only)
SECRET_KEY=dev-secret-key-not-for-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=./logs/app.log
EOF
}

# =============================================================================
# Backend Code Generation
# =============================================================================

create_main_app() {
    log_create "Creating main FastAPI application..."
    
    cat > "$PROJECT_ROOT/backend/app/main.py" << 'EOF'
"""
Self-Learning LLM Platform - Main Application
FastAPI application with self-improving language model capabilities
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import logging
from pathlib import Path

from .api import chat, feedback, embeddings, training
from .core.config import settings
from .core.database import engine, Base

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Self-Learning LLM Platform",
    description="An intelligent platform with self-improving language model capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["feedback"])
app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["embeddings"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting Self-Learning LLM Platform...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created")
    
    # Initialize services
    from .services.llm_service import llm_service
    from .services.embedding_service import embedding_service
    
    await llm_service.initialize()
    await embedding_service.initialize()
    
    logger.info("✅ Services initialized")
    logger.info(f"🌐 Server running on {settings.SERVER_HOST}:{settings.SERVER_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Self-Learning LLM Platform...")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Self-Learning LLM Platform</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .api-link { display: inline-block; margin: 10px; padding: 10px 20px; 
                       background: #007bff; color: white; text-decoration: none; 
                       border-radius: 5px; }
            .api-link:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Self-Learning LLM Platform</h1>
                <p>An intelligent platform with self-improving language model capabilities</p>
            </div>
            <div>
                <h2>API Documentation</h2>
                <a href="/docs" class="api-link">📚 Swagger UI</a>
                <a href="/redoc" class="api-link">📖 ReDoc</a>
            </div>
            <div>
                <h2>Quick Start</h2>
                <p>Use the following endpoints to interact with the platform:</p>
                <ul>
                    <li><strong>POST /api/v1/chat/</strong> - Chat with the AI</li>
                    <li><strong>POST /api/v1/feedback/</strong> - Provide feedback</li>
                    <li><strong>POST /api/v1/embeddings/</strong> - Generate embeddings</li>
                    <li><strong>POST /api/v1/training/start</strong> - Start training</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.DEBUG
    )
EOF
}

create_config_files() {
    log_create "Creating configuration files..."
    
    # Create __init__.py files
    touch "$PROJECT_ROOT/backend/__init__.py"
    touch "$PROJECT_ROOT/backend/app/__init__.py"
    touch "$PROJECT_ROOT/backend/app/api/__init__.py"
    touch "$PROJECT_ROOT/backend/app/core/__init__.py"
    touch "$PROJECT_ROOT/backend/app/models/__init__.py"
    touch "$PROJECT_ROOT/backend/app/services/__init__.py"
    touch "$PROJECT_ROOT/backend/app/utils/__init__.py"
    
    # Create config.py
    cat > "$PROJECT_ROOT/backend/app/core/config.py" << 'EOF'
"""
Configuration settings for the Self-Learning LLM Platform
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # Server Configuration
    SERVER_HOST: str = "localhost"
    SERVER_PORT: int = 8000
    DEBUG: bool = True
    
    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "self_learning_llm_dev"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "password"
    DATABASE_URL: Optional[str] = None
    
    # Model Configuration
    MODEL_CACHE_DIR: str = "./models"
    PHI2_MODEL_PATH: str = "./models/phi2"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Faiss Configuration
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    VECTOR_DIMENSION: int = 384
    
    # Security
    SECRET_KEY: str = "dev-secret-key-not-for-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "./logs/app.log"
    
    class Config:
        env_file = "config/.env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Auto-generate DATABASE_URL if not provided
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        
        # Ensure directories exist
        Path(self.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(self.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()
EOF

    # Create database.py
    cat > "$PROJECT_ROOT/backend/app/core/database.py" << 'EOF'
"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG  # Log SQL queries in debug mode
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
EOF
}

create_models() {
    log_create "Creating data models..."
    
    # Create database models
    cat > "$PROJECT_ROOT/backend/app/models/database.py" << 'EOF'
"""
Database models for the Self-Learning LLM Platform
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from ..core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=True)
    email = Column(String(100), unique=True, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    title = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    token_count = Column(Integer, default=0)
    processing_time_ms = Column(Integer, default=0)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    model_outputs = relationship("ModelOutput", back_populates="message")
    feedbacks = relationship("Feedback", back_populates="message")

class ModelOutput(Base):
    __tablename__ = "model_outputs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"))
    model_name = Column(String(100), default="phi2")
    model_version = Column(String(50), nullable=True)
    prompt_text = Column(Text, nullable=False)
    generated_text = Column(Text, nullable=False)
    confidence_score = Column(Float, default=0.0)
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=1024)
    actual_tokens = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    message = relationship("Message", back_populates="model_outputs")

class Feedback(Base):
    __tablename__ = "feedbacks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"))
    model_output_id = Column(UUID(as_uuid=True), ForeignKey("model_outputs.id"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    feedback_type = Column(String(20), nullable=False)  # positive, negative, neutral, correction
    rating = Column(Integer, nullable=True)  # 1-5
    feedback_text = Column(Text, nullable=True)
    corrected_response = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed = Column(Boolean, default=False)
    
    # Relationships
    message = relationship("Message", back_populates="feedbacks")

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=True)
    content = Column(Text, nullable=False)
    source_url = Column(String(500), nullable=True)
    document_type = Column(String(50), default="text")
    embedding_id = Column(UUID(as_uuid=True), nullable=True)
    tags = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)

class TrainingSession(Base):
    __tablename__ = "training_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_name = Column(String(200), nullable=True)
    base_model = Column(String(100), default="phi2")
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    training_data_size = Column(Integer, nullable=True)
    training_parameters = Column(JSON, default={})
    loss_history = Column(JSON, default=[])
    model_checkpoint_path = Column(String(500), nullable=True)
    performance_metrics = Column(JSON, default={})
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
EOF

    # Create schemas
    cat > "$PROJECT_ROOT/backend/app/models/schemas.py" << 'EOF'
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class FeedbackType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# User schemas
class UserBase(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None


class UserCreate(UserBase):
    username: str


class User(UserBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


# Conversation schemas
class ConversationBase(BaseModel):
    title: Optional[str] = None


class ConversationCreate(ConversationBase):
    user_id: UUID


class Conversation(ConversationBase):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


# Message schemas
class MessageBase(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class MessageCreate(MessageBase):
    conversation_id: UUID


class Message(MessageBase):
    id: UUID
    conversation_id: UUID
    created_at: datetime
    token_count: int
    processing_time_ms: int

    class Config:
        from_attributes = True


# Chat request/response schemas
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    use_rag: Optional[bool] = True


class ChatResponse(BaseModel):
    message_id: UUID
    conversation_id: UUID
    response: str
    processing_time_ms: int
    token_count: int
    confidence_score: Optional[float] = None
    rag_sources: Optional[List[Dict[str, Any]]] = []


# Model output schemas
class ModelOutputBase(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_name: str = "phi2"
    model_version: Optional[str] = None
    prompt_text: str
    generated_text: str
    confidence_score: Optional[float] = 0.0
    temperature: float = 0.7
    max_tokens: int = 1024
    actual_tokens: Optional[int] = None
    processing_time_ms: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = {}


class ModelOutputCreate(ModelOutputBase):
    message_id: UUID


class ModelOutput(ModelOutputBase):
    id: UUID
    message_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Feedback schemas
class FeedbackBase(BaseModel):
    feedback_type: FeedbackType
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    feedback_text: Optional[str] = None
    corrected_response: Optional[str] = None


class FeedbackCreate(FeedbackBase):
    model_config = {"protected_namespaces": ()}
    
    message_id: UUID
    model_output_id: Optional[UUID] = None
    user_id: Optional[UUID] = None


class Feedback(FeedbackBase):
    id: UUID
    message_id: UUID
    model_output_id: Optional[UUID]
    user_id: Optional[UUID]
    created_at: datetime
    processed: bool

    class Config:
        from_attributes = True


# Training session schemas
class TrainingSessionBase(BaseModel):
    session_name: Optional[str] = None
    base_model: str = "phi2"
    training_data_size: Optional[int] = None
    training_parameters: Optional[Dict[str, Any]] = {}


class TrainingSessionCreate(TrainingSessionBase):
    created_by: UUID


class TrainingSession(TrainingSessionBase):
    id: UUID
    start_time: datetime
    end_time: Optional[datetime]
    status: TrainingStatus
    loss_history: List[float]
    model_checkpoint_path: Optional[str]
    performance_metrics: Dict[str, Any]
    created_by: UUID

    class Config:
        from_attributes = True


# Search schemas
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = Field(default=5, ge=1, le=20)
    threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    content: str
    title: Optional[str] = None
    source_url: Optional[str] = None
    similarity_score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    processing_time_ms: int
EOF
}

create_api_endpoints() {
    log_create "Creating API endpoints..."
    
    # Chat API
    cat > "$PROJECT_ROOT/backend/app/api/chat.py" << 'EOF'
"""
Chat API endpoints for the Self-Learning LLM Platform
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import time
import uuid

from ..core.database import get_db
from ..models.schemas import ChatRequest, ChatResponse, MessageCreate, Message
from ..models.database import Message as DBMessage, Conversation as DBConversation
from ..services.llm_service import llm_service
from ..services.embedding_service import embedding_service

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Main chat endpoint - processes user messages and returns AI responses
    """
    start_time = time.time()
    
    try:
        # Create or get conversation
        if request.conversation_id:
            conversation = db.query(DBConversation).filter(
                DBConversation.id == request.conversation_id
            ).first()
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            # Create new conversation
            conversation = DBConversation(
                user_id=request.user_id,
                title=f"Chat {time.strftime('%Y-%m-%d %H:%M')}"
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        
        # Save user message
        user_message = DBMessage(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            token_count=len(request.message.split()),
            processing_time_ms=0
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        
        # Generate AI response
        ai_response = await llm_service.generate_response(
            prompt=request.message,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # RAG enhancement if requested
        rag_sources = []
        if request.use_rag:
            try:
                search_results = await embedding_service.search_similar(
                    query=request.message,
                    limit=3
                )
                if search_results:
                    context = "\n".join([result["content"] for result in search_results])
                    enhanced_prompt = f"Context: {context}\n\nUser: {request.message}\n\nAssistant:"
                    ai_response = await llm_service.generate_response(
                        prompt=enhanced_prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                    rag_sources = search_results
            except Exception as e:
                # Continue without RAG if it fails
                pass
        
        # Save AI message
        ai_message = DBMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=ai_response["text"],
            token_count=ai_response.get("token_count", 0),
            processing_time_ms=ai_response.get("processing_time_ms", 0),
            metadata={"confidence_score": ai_response.get("confidence_score", 0.0)}
        )
        db.add(ai_message)
        db.commit()
        db.refresh(ai_message)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ChatResponse(
            message_id=ai_message.id,
            conversation_id=conversation.id,
            response=ai_response["text"],
            processing_time_ms=processing_time,
            token_count=ai_response.get("token_count", 0),
            confidence_score=ai_response.get("confidence_score", 0.0),
            rag_sources=rag_sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.get("/conversations/{conversation_id}/messages", response_model=List[Message])
async def get_conversation_messages(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get all messages in a conversation"""
    messages = db.query(DBMessage).filter(
        DBMessage.conversation_id == conversation_id
    ).order_by(DBMessage.created_at).all()
    
    return messages

@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get conversation details"""
    conversation = db.query(DBConversation).filter(
        DBConversation.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation
EOF

    # Feedback API
    cat > "$PROJECT_ROOT/backend/app/api/feedback.py" << 'EOF'
"""
Feedback API endpoints for the Self-Learning LLM Platform
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uuid

from ..core.database import get_db
from ..models.schemas import FeedbackCreate, Feedback
from ..models.database import Feedback as DBFeedback
from ..services.training_service import training_service

router = APIRouter()

@router.post("/", response_model=Feedback)
async def create_feedback(
    feedback: FeedbackCreate,
    db: Session = Depends(get_db)
):
    """
    Submit feedback for a message/model output
    """
    try:
        # Create feedback record
        db_feedback = DBFeedback(**feedback.dict())
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        
        # Process feedback for training if it's a correction
        if feedback.feedback_type == "correction" and feedback.corrected_response:
            await training_service.add_training_example(
                original_response=feedback.feedback_text or "",
                corrected_response=feedback.corrected_response,
                feedback_id=db_feedback.id
            )
        
        return db_feedback
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create feedback: {str(e)}")

@router.get("/message/{message_id}", response_model=List[Feedback])
async def get_message_feedback(
    message_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get all feedback for a specific message"""
    feedback_list = db.query(DBFeedback).filter(
        DBFeedback.message_id == message_id
    ).all()
    
    return feedback_list

@router.get("/unprocessed", response_model=List[Feedback])
async def get_unprocessed_feedback(
    db: Session = Depends(get_db),
    limit: int = 100
):
    """Get unprocessed feedback for training"""
    feedback_list = db.query(DBFeedback).filter(
        DBFeedback.processed == False
    ).limit(limit).all()
    
    return feedback_list

@router.patch("/{feedback_id}/process")
async def mark_feedback_processed(
    feedback_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Mark feedback as processed"""
    feedback = db.query(DBFeedback).filter(DBFeedback.id == feedback_id).first()
    
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    feedback.processed = True
    db.commit()
    
    return {"message": "Feedback marked as processed"}
EOF

    # Embeddings API
    cat > "$PROJECT_ROOT/backend/app/api/embeddings.py" << 'EOF'
"""
Embeddings API endpoints for the Self-Learning LLM Platform
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from ..core.database import get_db
from ..models.schemas import SearchRequest, SearchResponse
from ..services.embedding_service import embedding_service

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def search_embeddings(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    """
    Search for similar content using embeddings
    """
    try:
        results = await embedding_service.search_similar(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold
        )
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            processing_time_ms=100  # Placeholder
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/embed")
async def create_embedding(
    content: str,
    db: Session = Depends(get_db)
):
    """
    Create embeddings for given content
    """
    try:
        embedding = await embedding_service.create_embedding(content)
        return {"embedding_id": embedding["id"], "dimension": len(embedding["vector"])}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding creation failed: {str(e)}")

@router.post("/add-knowledge")
async def add_knowledge(
    title: str,
    content: str,
    source_url: str = None,
    db: Session = Depends(get_db)
):
    """
    Add content to the knowledge base
    """
    try:
        result = await embedding_service.add_to_knowledge_base(
            title=title,
            content=content,
            source_url=source_url
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add knowledge: {str(e)}")
EOF

    # Training API
    cat > "$PROJECT_ROOT/backend/app/api/training.py" << 'EOF'
"""
Training API endpoints for the Self-Learning LLM Platform
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import uuid

from ..core.database import get_db
from ..models.schemas import TrainingSessionCreate, TrainingSession
from ..models.database import TrainingSession as DBTrainingSession
from ..services.training_service import training_service

router = APIRouter()

@router.post("/start", response_model=TrainingSession)
async def start_training_session(
    session_data: TrainingSessionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start a new training session
    """
    try:
        # Create training session record
        db_session = DBTrainingSession(**session_data.dict())
        db_session.status = "pending"
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        # Start training in background
        background_tasks.add_task(
            training_service.start_training_session,
            session_id=db_session.id
        )
        
        return db_session
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/sessions", response_model=List[TrainingSession])
async def get_training_sessions(
    db: Session = Depends(get_db),
    limit: int = 10
):
    """Get list of training sessions"""
    sessions = db.query(DBTrainingSession).order_by(
        DBTrainingSession.start_time.desc()
    ).limit(limit).all()
    
    return sessions

@router.get("/sessions/{session_id}", response_model=TrainingSession)
async def get_training_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get specific training session details"""
    session = db.query(DBTrainingSession).filter(
        DBTrainingSession.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return session

@router.post("/sessions/{session_id}/stop")
async def stop_training_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Stop a running training session"""
    try:
        await training_service.stop_training_session(session_id)
        return {"message": "Training session stopped"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")

@router.get("/metrics/{session_id}")
async def get_training_metrics(
    session_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get training metrics for a session"""
    session = db.query(DBTrainingSession).filter(
        DBTrainingSession.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return {
        "session_id": session_id,
        "loss_history": session.loss_history,
        "performance_metrics": session.performance_metrics,
        "status": session.status
    }
EOF
}

create_services() {
    log_create "Creating service modules..."
    
    # LLM Service
    cat > "$PROJECT_ROOT/backend/app/services/llm_service.py" << 'EOF'
"""
LLM Service for the Self-Learning LLM Platform
Handles model loading, inference, and response generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging
from typing import Dict, Any, Optional
from ..core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "microsoft/phi-2"
        self.initialized = False
    
    async def initialize(self):
        """Initialize the LLM model and tokenizer"""
        try:
            logger.info("Initializing LLM service...")
            
            # Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info(f"Loading model {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                cache_dir=settings.MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.initialized = True
            
            logger.info(f"✅ LLM service initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate a response using the LLM"""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare the prompt
            formatted_prompt = f"Human: {prompt}\n\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            generated_text = full_response[len(formatted_prompt):].strip()
            
            # Calculate metrics
            processing_time_ms = int((time.time() - start_time) * 1000)
            token_count = len(self.tokenizer.encode(generated_text))
            
            # Simple confidence score based on response length and coherence
            confidence_score = min(1.0, len(generated_text) / 100) * 0.8
            
            return {
                "text": generated_text,
                "token_count": token_count,
                "processing_time_ms": processing_time_ms,
                "confidence_score": confidence_score,
                "model_name": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "text": "I apologize, but I encountered an error while processing your request. Please try again.",
                "token_count": 0,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "confidence_score": 0.0,
                "model_name": self.model_name,
                "error": str(e)
            }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "initialized": self.initialized,
            "model_size": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }

# Global service instance
llm_service = LLMService()
EOF

    # Embedding Service
    cat > "$PROJECT_ROOT/backend/app/services/embedding_service.py" << 'EOF'
"""
Embedding Service for the Self-Learning LLM Platform
Handles text embeddings, similarity search, and knowledge base management
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.index = None
        self.knowledge_base = []
        self.initialized = False
        self.dimension = settings.VECTOR_DIMENSION
        self.index_path = settings.FAISS_INDEX_PATH
        self.kb_path = f"{self.index_path}_kb.json"
    
    async def initialize(self):
        """Initialize the embedding service"""
        try:
            logger.info("Initializing embedding service...")
            
            # Load sentence transformer model
            logger.info(f"Loading embedding model: {settings.EMBEDDINGS_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDINGS_MODEL)
            
            # Update dimension based on model
            test_embedding = self.model.encode(["test"])
            self.dimension = len(test_embedding[0])
            
            # Initialize or load FAISS index
            await self._load_or_create_index()
            
            self.initialized = True
            logger.info(f"✅ Embedding service initialized with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    async def _load_or_create_index(self):
        """Load existing FAISS index or create a new one"""
        index_file = f"{self.index_path}.index"
        
        if Path(index_file).exists() and Path(self.kb_path).exists():
            try:
                # Load existing index
                self.index = faiss.read_index(index_file)
                
                # Load knowledge base
                with open(self.kb_path, 'r') as f:
                    self.knowledge_base = json.load(f)
                
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
        
        # Create new index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.knowledge_base = []
        logger.info("Created new FAISS index")
    
    async def create_embedding(self, text: str) -> Dict[str, Any]:
        """Create embedding for given text"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Generate embedding
            embedding = self.model.encode([text])
            vector = embedding[0].astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(vector.reshape(1, -1))
            
            # Create hash for deduplication
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            return {
                "id": text_hash,
                "vector": vector,
                "text": text,
                "dimension": len(vector)
            }
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    async def add_to_knowledge_base(
        self,
        title: str,
        content: str,
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add content to the knowledge base"""
        try:
            # Create embedding
            embedding_data = await self.create_embedding(content)
            
            # Prepare knowledge base entry
            kb_entry = {
                "id": embedding_data["id"],
                "title": title,
                "content": content,
                "source_url": source_url,
                "metadata": metadata or {},
                "vector_index": len(self.knowledge_base)
            }
            
            # Add to FAISS index
            vector = embedding_data["vector"].reshape(1, -1)
            self.index.add(vector)
            
            # Add to knowledge base
            self.knowledge_base.append(kb_entry)
            
            # Save to disk
            await self._save_index()
            
            logger.info(f"Added '{title}' to knowledge base")
            
            return {
                "id": embedding_data["id"],
                "title": title,
                "vector_index": kb_entry["vector_index"]
            }
            
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {e}")
            raise
    
    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar content in the knowledge base"""
        if not self.initialized:
            await self.initialize()
        
        if self.index.ntotal == 0:
            return []
        
        try:
            # Create query embedding
            query_embedding = await self.create_embedding(query)
            query_vector = query_embedding["vector"].reshape(1, -1)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_vector, min(limit, self.index.ntotal))
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and idx < len(self.knowledge_base):
                    kb_entry = self.knowledge_base[idx]
                    results.append({
                        "content": kb_entry["content"],
                        "title": kb_entry["title"],
                        "source_url": kb_entry["source_url"],
                        "similarity_score": float(score),
                        "metadata": kb_entry["metadata"]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def _save_index(self):
        """Save FAISS index and knowledge base to disk"""
        try:
            # Ensure directory exists
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            # Save knowledge base
            with open(self.kb_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            
            logger.debug("Saved index and knowledge base to disk")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        return {
            "initialized": self.initialized,
            "model_name": settings.EMBEDDINGS_MODEL,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal if self.index else 0,
            "knowledge_base_size": len(self.knowledge_base)
        }

# Global service instance
embedding_service = EmbeddingService()
EOF

    # Training Service
    cat > "$PROJECT_ROOT/backend/app/services/training_service.py" << 'EOF'
"""
Training Service for the Self-Learning LLM Platform
Handles model training, fine-tuning, and learning from feedback
"""

import torch
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from ..core.config import settings

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.training_data = []
        self.active_sessions = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize the training service"""
        try:
            logger.info("Initializing training service...")
            
            # Load existing training data if available
            await self._load_training_data()
            
            self.initialized = True
            logger.info("✅ Training service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize training service: {e}")
            raise
    
    async def add_training_example(
        self,
        original_response: str,
        corrected_response: str,
        feedback_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a training example from feedback"""
        try:
            training_example = {
                "id": feedback_id or f"example_{len(self.training_data)}",
                "original": original_response,
                "corrected": corrected_response,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            self.training_data.append(training_example)
            
            # Save to disk
            await self._save_training_data()
            
            logger.info(f"Added training example: {training_example['id']}")
            
        except Exception as e:
            logger.error(f"Error adding training example: {e}")
            raise
    
    async def start_training_session(self, session_id: str):
        """Start a training session"""
        try:
            logger.info(f"Starting training session: {session_id}")
            
            # Mock training process (replace with actual training logic)
            self.active_sessions[session_id] = {
                "status": "running",
                "start_time": datetime.now(),
                "progress": 0.0,
                "loss_history": []
            }
            
            # Simulate training progress
            for epoch in range(5):
                await asyncio.sleep(10)  # Simulate training time
                
                # Mock loss calculation
                loss = 1.0 - (epoch * 0.15) + (torch.rand(1).item() * 0.1)
                
                self.active_sessions[session_id]["progress"] = (epoch + 1) / 5
                self.active_sessions[session_id]["loss_history"].append(loss)
                
                logger.info(f"Session {session_id} - Epoch {epoch + 1}/5, Loss: {loss:.4f}")
            
            # Mark as completed
            self.active_sessions[session_id]["status"] = "completed"
            self.active_sessions[session_id]["end_time"] = datetime.now()
            
            logger.info(f"Training session {session_id} completed")
            
        except Exception as e:
            logger.error(f"Error in training session {session_id}: {e}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "failed"
                self.active_sessions[session_id]["error"] = str(e)
    
    async def stop_training_session(self, session_id: str):
        """Stop a running training session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "stopped"
            logger.info(f"Training session {session_id} stopped")
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the status of a training session"""
        return self.active_sessions.get(session_id, {"status": "not_found"})
    
    async def create_training_dataset(self) -> Dataset:
        """Create a training dataset from collected examples"""
        if not self.training_data:
            return Dataset.from_dict({"input": [], "output": []})
        
        inputs = []
        outputs = []
        
        for example in self.training_data:
            # Create training pairs
            inputs.append(f"Original: {example['original']}")
            outputs.append(example['corrected'])
        
        return Dataset.from_dict({
            "input": inputs,
            "output": outputs
        })
    
    async def _load_training_data(self):
        """Load training data from disk"""
        training_file = Path(settings.MODEL_CACHE_DIR) / "training_data.json"
        
        if training_file.exists():
            try:
                with open(training_file, 'r') as f:
                    self.training_data = json.load(f)
                logger.info(f"Loaded {len(self.training_data)} training examples")
            except Exception as e:
                logger.warning(f"Failed to load training data: {e}")
                self.training_data = []
        else:
            self.training_data = []
    
    async def _save_training_data(self):
        """Save training data to disk"""
        training_file = Path(settings.MODEL_CACHE_DIR) / "training_data.json"
        training_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(training_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            logger.debug("Saved training data to disk")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    async def get_training_stats(self) -> Dict[str, Any]:
        """Get training service statistics"""
        return {
            "initialized": self.initialized,
            "total_examples": len(self.training_data),
            "active_sessions": len([s for s in self.active_sessions.values() if s["status"] == "running"]),
            "completed_sessions": len([s for s in self.active_sessions.values() if s["status"] == "completed"])
        }

# Global service instance
training_service = TrainingService()
EOF
}

create_scripts() {
    log_create "Creating utility scripts..."
    
    # Run local script
    cat > "$PROJECT_ROOT/scripts/run_local.py" << 'EOF'
#!/usr/bin/env python3
"""
Local Development Runner
Script to run the Self-Learning LLM Platform locally for development and testing
"""

import os
import sys
import subprocess
import argparse
import logging
import time
import signal
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalRunner:
    def __init__(self, host: str = "localhost", port: int = 8000, reload: bool = True):
        self.host = host
        self.port = port
        self.reload = reload
        self.processes = []
        self.project_root = project_root
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        try:
            logger.info("Checking dependencies...")
            
            # Check if we're in a virtual environment
            if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
                logger.warning("Not running in a virtual environment. Consider using 'python -m venv venv' and activating it.")
            
            # Try importing key dependencies
            try:
                import fastapi
                import uvicorn
                import transformers
                import faiss
                import sqlalchemy
                import psycopg2
                logger.info("✅ All required dependencies found")
                return True
            except ImportError as e:
                logger.error(f"❌ Missing dependency: {e}")
                logger.error("Please install requirements with: pip install -r requirements.txt")
                return False
            
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False
    
    def check_environment(self) -> bool:
        """Check environment configuration"""
        try:
            logger.info("Checking environment configuration...")
            
            env_file = self.project_root / "config" / ".env"
            if not env_file.exists():
                logger.warning(f"Environment file not found: {env_file}")
                logger.info("Creating a basic environment file...")
                
                # Create basic .env file
                env_file.parent.mkdir(exist_ok=True)
                env_content = f"""# Database Configuration (for local development)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=self_learning_llm_dev
DB_USER=postgres
DB_PASSWORD=password
DATABASE_URL=postgresql://postgres:password@localhost:5432/self_learning_llm_dev

# Server Configuration
SERVER_HOST={self.host}
SERVER_PORT={self.port}
DEBUG=true

# Model Configuration
MODEL_CACHE_DIR=./models
PHI2_MODEL_PATH=./models/phi2
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Faiss Configuration
FAISS_INDEX_PATH=./data/faiss_index
VECTOR_DIMENSION=384

# Security (development only)
SECRET_KEY=dev-secret-key-not-for-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=./logs/app.log
"""
                env_file.write_text(env_content)
                logger.info(f"Created basic environment file: {env_file}")
            
            logger.info("✅ Environment configuration ready")
            return True
            
        except Exception as e:
            logger.error(f"Environment check failed: {e}")
            return False
    
    def setup_directories(self) -> bool:
        """Create necessary directories"""
        try:
            logger.info("Setting up directories...")
            
            directories = [
                "models",
                "data",
                "logs",
                "config"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                logger.info(f"Directory ready: {dir_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            return False
    
    def download_model_if_needed(self) -> bool:
        """Download Phi-2 model if not present"""
        try:
            model_path = self.project_root / "models" / "phi2"
            
            if model_path.exists() and any(model_path.iterdir()):
                logger.info("✅ Phi-2 model already downloaded")
                return True
            
            logger.info("✅ Phi-2 model will be downloaded automatically on first use")
            return True
            
        except Exception as e:
            logger.error(f"Model download check failed: {e}")
            return False
    
    def start_server(self) -> bool:
        """Start the FastAPI server"""
        try:
            logger.info(f"Starting server on {self.host}:{self.port}...")
            
            # Change to project root
            os.chdir(self.project_root)
            
            # Build uvicorn command
            cmd = [
                sys.executable, "-m", "uvicorn",
                "backend.app.main:app",
                "--host", self.host,
                "--port", str(self.port)
            ]
            
            if self.reload:
                cmd.append("--reload")
            
            # Start the server
            process = subprocess.Popen(cmd)
            self.processes.append(process)
            
            logger.info("✅ Server started successfully!")
            logger.info(f"🌐 Access the application at: http://{self.host}:{self.port}")
            logger.info(f"📚 API documentation at: http://{self.host}:{self.port}/docs")
            logger.info("Press Ctrl+C to stop the server")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            # Wait for processes to finish
            for process in self.processes:
                process.wait()
        except KeyboardInterrupt:
            logger.info("Shutdown signal received...")
            self.stop_all_processes()
    
    def stop_all_processes(self):
        """Stop all running processes"""
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.warning(f"Error stopping process: {e}")
        
        logger.info("All processes stopped")
    
    def run(self) -> bool:
        """Run the complete local setup"""
        try:
            logger.info("🚀 Starting Self-Learning LLM Platform locally...")
            
            if not self.check_dependencies():
                return False
            
            if not self.check_environment():
                return False
            
            if not self.setup_directories():
                return False
            
            if not self.download_model_if_needed():
                return False
            
            if not self.start_server():
                return False
            
            self.wait_for_shutdown()
            return True
            
        except Exception as e:
            logger.error(f"Local run failed: {e}")
            return False
        finally:
            self.stop_all_processes()


def main():
    parser = argparse.ArgumentParser(description="Run Self-Learning LLM Platform locally")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and environment")
    
    args = parser.parse_args()
    
    runner = LocalRunner(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )
    
    if args.check_only:
        logger.info("Running environment check only...")
        if runner.check_dependencies() and runner.check_environment():
            logger.info("✅ Environment check passed!")
            return 0
        else:
            logger.error("❌ Environment check failed!")
            return 1
    
    if runner.run():
        logger.info("✅ Local run completed successfully!")
        return 0
    else:
        logger.error("❌ Local run failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x "$PROJECT_ROOT/scripts/run_local.py"
}

create_additional_files() {
    log_create "Creating additional configuration files..."
    
    # Create .gitignore
    cat > "$PROJECT_ROOT/.gitignore" << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# Environment Variables
.env
.env.local
.env.production

# Models and Data
models/
data/faiss_index*
logs/
*.log

# Database
*.db
*.sqlite3

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/
EOF

    # Create README.md
    cat > "$PROJECT_ROOT/README.md" << 'EOF'
# Self-Learning LLM Platform

An intelligent platform with self-improving language model capabilities, built with FastAPI and powered by Microsoft Phi-2.

## Features

- 🤖 **Intelligent Chat Interface**: Interactive AI conversations with context awareness
- 📚 **Knowledge Base Integration**: RAG (Retrieval-Augmented Generation) support
- 🔄 **Self-Learning Capabilities**: Learn from user feedback and improve responses
- 🎯 **Vector Search**: Semantic search using sentence transformers and FAISS
- 📊 **Training Management**: Fine-tuning and model improvement workflows
- 🔒 **Secure API**: JWT authentication and rate limiting
- 📈 **Performance Monitoring**: Comprehensive logging and metrics

## Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for optimal performance)
- CUDA-compatible GPU (optional, for faster inference)

### Installation and Setup

1. **Automated Setup** (Recommended):
   ```bash
   # Make the setup script executable
   chmod +x create_llm_platform.sh
   
   # Run the complete setup
   ./create_llm_platform.sh
   ```

2. **Manual Setup**:
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate
   
   # Install dependencies
   cd self-learning-llm
   pip install -r requirements.txt
   
   # Run the application
   python scripts/run_local.py
   ```

### Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## API Endpoints

### Chat
- `POST /api/v1/chat/` - Send a message to the AI
- `GET /api/v1/chat/conversations/{conversation_id}/messages` - Get conversation history

### Feedback
- `POST /api/v1/feedback/` - Submit feedback for model improvement
- `GET /api/v1/feedback/unprocessed` - Get unprocessed feedback

### Embeddings & Search
- `POST /api/v1/embeddings/search` - Semantic search in knowledge base
- `POST /api/v1/embeddings/add-knowledge` - Add content to knowledge base

### Training
- `POST /api/v1/training/start` - Start a training session
- `GET /api/v1/training/sessions` - List training sessions

## Configuration

The application uses environment variables for configuration. See `config/.env` for available options:

- **Database**: PostgreSQL connection settings
- **Models**: Model cache directories and configurations
- **Security**: JWT secret keys and authentication settings
- **Logging**: Log levels and file paths

## Development

### Project Structure

```
self-learning-llm/
├── backend/
│   └── app/
│       ├── api/          # API endpoints
│       ├── core/         # Configuration and database
│       ├── models/       # Data models and schemas
│       ├── services/     # Business logic services
│       └── utils/        # Utility functions
├── scripts/              # Utility scripts
├── config/               # Configuration files
├── data/                 # Data storage
├── logs/                 # Application logs
├── models/               # Model storage
└── tests/                # Test files
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black backend/
flake8 backend/
```

## Model Information

- **Base Model**: Microsoft Phi-2 (2.7B parameters)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: FAISS for similarity search
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient training

## Performance Optimization

- **CPU Mode**: Automatic fallback for systems without CUDA
- **Model Caching**: Persistent model storage for faster startup
- **Vector Indexing**: Optimized FAISS indices for fast retrieval
- **Async Processing**: Non-blocking API operations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation at `/docs`
- Review the API documentation at `/api/docs`

## Roadmap

- [ ] Multi-modal support (images, audio)
- [ ] Advanced fine-tuning strategies
- [ ] Distributed training support
- [ ] Enhanced security features
- [ ] Real-time collaboration features
- [ ] Plugin system for extensions
EOF

    # Create Dockerfile
    cat > "$PROJECT_ROOT/Dockerfile" << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data logs config

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    # Create docker-compose.yml
    cat > "$PROJECT_ROOT/docker-compose.yml" << 'EOF'
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/self_learning_llm
    depends_on:
      - db
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=self_learning_llm
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
EOF
}

setup_virtual_environment_and_install() {
    if [ "$SKIP_INSTALL" = "true" ]; then
        log_info "Skipping dependency installation (SKIP_INSTALL=true)"
        return 0
    fi
    
    log_step "Setting up virtual environment and installing dependencies..."
    
    # Create virtual environment
    if [ ! -d "$VENV_PATH" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    log_info "Installing dependencies..."
    cd "$PROJECT_ROOT"
    pip install -r requirements.txt
    
    log_success "Dependencies installed successfully"
}

start_application() {
    log_step "Starting the Self-Learning LLM Platform..."
    
    cd "$PROJECT_ROOT"
    source "$VENV_PATH/bin/activate"
    
    log_success "🚀 Self-Learning LLM Platform is ready!"
    log_info "🌐 Access the application at: http://$HOST:$PORT"
    log_info "📚 API documentation at: http://$HOST:$PORT/docs"
    log_info "Press Ctrl+C to stop the server"
    
    # Start the server
    python scripts/run_local.py --host "$HOST" --port "$PORT"
}

show_usage() {
    echo "Self-Learning LLM Platform - Complete Code Generator and Setup"
    echo ""
    echo "Usage: $0 [HOST] [PORT] [SKIP_INSTALL]"
    echo ""
    echo "Arguments:"
    echo "  HOST          Server host (default: localhost)"
    echo "  PORT          Server port (default: 8000)"
    echo "  SKIP_INSTALL  Skip dependency installation (true/false, default: false)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Setup and start on localhost:8000"
    echo "  $0 0.0.0.0 8080      # Setup and start on 0.0.0.0:8080"
    echo "  $0 localhost 8000 true  # Setup without installing dependencies"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    log_success "🚀 Self-Learning LLM Platform - Complete Setup Script"
    log_info "======================================================="
    log_info "This script will create the complete Self-Learning LLM platform"
    log_info "Target Host: $HOST"
    log_info "Target Port: $PORT"
    log_info "Skip Install: $SKIP_INSTALL"
    echo ""
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Execute setup steps
    create_directory_structure
    create_requirements_txt
    create_env_file
    create_main_app
    create_config_files
    create_models
    create_api_endpoints
    create_services
    create_scripts
    create_additional_files
    
    if [ "$SKIP_INSTALL" != "true" ]; then
        setup_virtual_environment_and_install
    fi
    
    log_success "✅ Self-Learning LLM Platform setup completed!"
    echo ""
    
    # Ask if user wants to start the application
    echo "Would you like to start the application now? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        start_application
    else
        log_info "Setup complete! To start the application later, run:"
        log_info "  cd self-learning-llm"
        log_info "  source ../.venv/bin/activate"
        log_info "  python scripts/run_local.py"
    fi
}

# Run main function
main "$@"
EOF
}

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Create bash script that generates complete code structure", "status": "completed", "priority": "high"}, {"id": "2", "content": "Generate all Python files and directory structure", "status": "completed", "priority": "high"}, {"id": "3", "content": "Include requirements.txt and configuration files", "status": "completed", "priority": "high"}, {"id": "4", "content": "Test the complete setup script", "status": "in_progress", "priority": "medium"}]