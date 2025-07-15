#!/usr/bin/env python3
"""
Rebuild the entire Faiss index from database
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy.orm import Session
from core.database import SessionLocal
from core.vector_store import vector_store

def main():
    """Rebuild the Faiss index from database"""
    print("Starting index rebuild...")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Rebuild index
        vector_store.rebuild_index(db)
        
        # Get final stats
        stats = vector_store.get_stats()
        print(f"\nRebuild complete!")
        print(f"Total embeddings in index: {stats['total_embeddings']}")
        print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
        print(f"Content type distribution: {stats['content_types']}")
        
    except Exception as e:
        print(f"Error during rebuild: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()