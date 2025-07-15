import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import hashlib
import os
from ..utils.config import settings

class EmbeddingManager:
    """Handles text embedding generation using sentence-transformers"""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.embedding_model_name
        self.dimensions = settings.embedding_dimensions
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(settings.phi2_model_path, exist_ok=True)
            
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Update dimensions based on actual model
            sample_embedding = self.model.encode("test")
            self.dimensions = len(sample_embedding)
            
            print(f"Embedding model loaded successfully. Dimensions: {self.dimensions}")
            
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Normalize text
            text = text.strip()
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Ensure correct dimensions
            if len(embedding) != self.dimensions:
                raise ValueError(f"Embedding dimension mismatch: expected {self.dimensions}, got {len(embedding)}")
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        try:
            # Normalize texts
            texts = [text.strip() for text in texts if text and text.strip()]
            
            if not texts:
                raise ValueError("No valid texts provided")
            
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Ensure correct shape
            if embeddings.shape[1] != self.dimensions:
                raise ValueError(f"Embedding dimension mismatch: expected {self.dimensions}, got {embeddings.shape[1]}")
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def generate_embedding_id(self, text: str, content_type: str = "text") -> str:
        """Generate a unique ID for an embedding"""
        # Create a hash of the text and content type
        content = f"{content_type}:{text}"
        hash_object = hashlib.md5(content.encode())
        return f"emb_{hash_object.hexdigest()}"
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure similarity is in range [-1, 1]
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Get information about the current embedding model"""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "is_loaded": self.model is not None
        }

# Global embedding manager instance
embedding_manager = EmbeddingManager()