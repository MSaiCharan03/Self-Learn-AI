import os
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/llm_platform"
    
    # Security
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Model paths
    phi2_model_path: str = "./data/models/phi-2"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # Faiss settings
    faiss_index_path: str = "./data/faiss_index"
    embedding_dimensions: int = 384
    
    # External APIs
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 9090
    debug: bool = False
    
    # CORS settings
    cors_origins: list = ["http://localhost:9090", "http://127.0.0.1:9090", "http://localhost:9000", "http://127.0.0.1:9000"]
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./data/logs/app.log"
    
    class Config:
        env_file = ".env"

settings = Settings()