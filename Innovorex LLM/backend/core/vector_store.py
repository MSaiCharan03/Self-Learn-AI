import faiss
import numpy as np
import os
import pickle
import json
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from ..models.conversation import VectorEmbedding
from ..utils.config import settings
from .embeddings import embedding_manager

class FaissVectorStore:
    """Faiss-based vector storage and similarity search"""
    
    def __init__(self):
        self.index = None
        self.index_path = settings.faiss_index_path
        self.dimensions = settings.embedding_dimensions
        self.index_file = os.path.join(self.index_path, "faiss_index.bin")
        self.metadata_file = os.path.join(self.index_path, "metadata.json")
        self.id_mapping_file = os.path.join(self.index_path, "id_mapping.pkl")
        
        # In-memory mappings
        self.id_to_idx = {}  # embedding_id -> faiss_index
        self.idx_to_id = {}  # faiss_index -> embedding_id
        self.metadata = {}   # embedding_id -> metadata
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load existing Faiss index"""
        try:
            # Create index directory
            os.makedirs(self.index_path, exist_ok=True)
            
            # Try to load existing index
            if self._load_index():
                print(f"Loaded existing Faiss index with {self.index.ntotal} vectors")
            else:
                print("Creating new Faiss index")
                self._create_new_index()
                
        except Exception as e:
            print(f"Error initializing Faiss index: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new Faiss index"""
        try:
            # Use IndexFlatIP (Inner Product) for cosine similarity
            # Note: For cosine similarity, vectors should be normalized
            self.index = faiss.IndexFlatIP(self.dimensions)
            
            # Reset mappings
            self.id_to_idx = {}
            self.idx_to_id = {}
            self.metadata = {}
            
            print(f"Created new Faiss index with {self.dimensions} dimensions")
            
        except Exception as e:
            print(f"Error creating new Faiss index: {e}")
            raise
    
    def _load_index(self) -> bool:
        """Load existing index and metadata"""
        try:
            if not os.path.exists(self.index_file):
                return False
            
            # Load Faiss index
            self.index = faiss.read_index(self.index_file)
            
            # Load ID mappings
            if os.path.exists(self.id_mapping_file):
                with open(self.id_mapping_file, 'rb') as f:
                    data = pickle.load(f)
                    self.id_to_idx = data.get('id_to_idx', {})
                    self.idx_to_id = data.get('idx_to_id', {})
            
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def _save_index(self):
        """Save index and metadata to disk"""
        try:
            # Save Faiss index
            faiss.write_index(self.index, self.index_file)
            
            # Save ID mappings
            with open(self.id_mapping_file, 'wb') as f:
                pickle.dump({
                    'id_to_idx': self.id_to_idx,
                    'idx_to_id': self.idx_to_id
                }, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving index: {e}")
            raise
    
    def add_embedding(
        self, 
        embedding_id: str, 
        embedding: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Add a single embedding to the index"""
        try:
            if embedding_id in self.id_to_idx:
                print(f"Embedding {embedding_id} already exists, updating...")
                return self.update_embedding(embedding_id, embedding, metadata)
            
            # Normalize embedding for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.reshape(1, -1).astype(np.float32)
            
            # Add to Faiss index
            current_idx = self.index.ntotal
            self.index.add(embedding)
            
            # Update mappings
            self.id_to_idx[embedding_id] = current_idx
            self.idx_to_id[current_idx] = embedding_id
            self.metadata[embedding_id] = metadata
            
            # Save periodically
            if self.index.ntotal % 100 == 0:
                self._save_index()
            
            return True
            
        except Exception as e:
            print(f"Error adding embedding {embedding_id}: {e}")
            return False
    
    def add_embeddings(
        self, 
        embedding_ids: List[str], 
        embeddings: np.ndarray, 
        metadata_list: List[Dict[str, Any]]
    ) -> bool:
        """Add multiple embeddings to the index"""
        try:
            if len(embedding_ids) != len(embeddings) != len(metadata_list):
                raise ValueError("Mismatched lengths of IDs, embeddings, and metadata")
            
            # Filter out existing embeddings
            new_ids = []
            new_embeddings = []
            new_metadata = []
            
            for i, embedding_id in enumerate(embedding_ids):
                if embedding_id not in self.id_to_idx:
                    new_ids.append(embedding_id)
                    new_embeddings.append(embeddings[i])
                    new_metadata.append(metadata_list[i])
            
            if not new_ids:
                print("No new embeddings to add")
                return True
            
            # Normalize embeddings for cosine similarity
            new_embeddings = np.array(new_embeddings, dtype=np.float32)
            norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            new_embeddings = new_embeddings / norms
            
            # Add to Faiss index
            start_idx = self.index.ntotal
            self.index.add(new_embeddings)
            
            # Update mappings
            for i, embedding_id in enumerate(new_ids):
                current_idx = start_idx + i
                self.id_to_idx[embedding_id] = current_idx
                self.idx_to_id[current_idx] = embedding_id
                self.metadata[embedding_id] = new_metadata[i]
            
            # Save index
            self._save_index()
            
            print(f"Added {len(new_ids)} new embeddings to index")
            return True
            
        except Exception as e:
            print(f"Error adding embeddings: {e}")
            return False
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar embeddings"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            top_k = min(top_k, self.index.ntotal)
            similarities, indices = self.index.search(query_embedding, top_k)
            
            # Process results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity < min_similarity:
                    continue
                    
                embedding_id = self.idx_to_id.get(idx)
                if embedding_id:
                    metadata = self.metadata.get(embedding_id, {})
                    results.append((embedding_id, float(similarity), metadata))
            
            return results
            
        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return []
    
    def update_embedding(
        self, 
        embedding_id: str, 
        embedding: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Update an existing embedding"""
        try:
            if embedding_id not in self.id_to_idx:
                return self.add_embedding(embedding_id, embedding, metadata)
            
            # For IndexFlatIP, we need to rebuild the index for updates
            # This is a limitation of Faiss IndexFlat
            idx = self.id_to_idx[embedding_id]
            
            # Update metadata
            self.metadata[embedding_id] = metadata
            
            # For now, we'll just update metadata and note that 
            # embedding updates require index rebuild
            print(f"Updated metadata for {embedding_id}. Note: Embedding vector updates require index rebuild.")
            
            return True
            
        except Exception as e:
            print(f"Error updating embedding {embedding_id}: {e}")
            return False
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding (requires index rebuild)"""
        try:
            if embedding_id not in self.id_to_idx:
                return False
            
            # Remove from mappings
            idx = self.id_to_idx[embedding_id]
            del self.id_to_idx[embedding_id]
            del self.idx_to_id[idx]
            del self.metadata[embedding_id]
            
            # Note: Faiss IndexFlat doesn't support deletion
            # For production, consider using IndexIVF or rebuild periodically
            print(f"Marked {embedding_id} for deletion. Index rebuild required for physical removal.")
            
            return True
            
        except Exception as e:
            print(f"Error deleting embedding {embedding_id}: {e}")
            return False
    
    def rebuild_index(self, db: Session):
        """Rebuild the entire index from database"""
        try:
            print("Rebuilding Faiss index from database...")
            
            # Get all embeddings from database
            embeddings = db.query(VectorEmbedding).all()
            
            if not embeddings:
                print("No embeddings found in database")
                return
            
            # Create new index
            self._create_new_index()
            
            # Process embeddings in batches
            batch_size = 1000
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                
                embedding_ids = []
                embedding_vectors = []
                metadata_list = []
                
                for emb in batch:
                    try:
                        # Generate embedding vector from text
                        vector = embedding_manager.generate_embedding(emb.content_text)
                        
                        embedding_ids.append(emb.embedding_id)
                        embedding_vectors.append(vector)
                        metadata_list.append({
                            "content_type": emb.content_type,
                            "content_id": str(emb.content_id),
                            "model_name": emb.model_name,
                            "created_at": emb.created_at.isoformat(),
                            "metadata": emb.metadata
                        })
                        
                    except Exception as e:
                        print(f"Error processing embedding {emb.embedding_id}: {e}")
                        continue
                
                if embedding_vectors:
                    self.add_embeddings(embedding_ids, np.array(embedding_vectors), metadata_list)
            
            # Save final index
            self._save_index()
            
            print(f"Index rebuilt with {self.index.ntotal} embeddings")
            
        except Exception as e:
            print(f"Error rebuilding index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_embeddings": self.index.ntotal if self.index else 0,
            "dimensions": self.dimensions,
            "index_type": "IndexFlatIP",
            "memory_usage_mb": self.index.ntotal * self.dimensions * 4 / (1024 * 1024) if self.index else 0,
            "content_types": self._get_content_type_stats()
        }
    
    def _get_content_type_stats(self) -> Dict[str, int]:
        """Get statistics by content type"""
        stats = {}
        for embedding_id, metadata in self.metadata.items():
            content_type = metadata.get("content_type", "unknown")
            stats[content_type] = stats.get(content_type, 0) + 1
        return stats

# Global vector store instance
vector_store = FaissVectorStore()