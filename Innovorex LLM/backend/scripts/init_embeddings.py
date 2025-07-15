#!/usr/bin/env python3
"""
Initialize embeddings for existing data in the database
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy.orm import Session
from core.database import SessionLocal
from utils.helpers import batch_process_embeddings
from core.vector_store import vector_store

async def main():
    """Initialize embeddings for existing data"""
    print("Starting embedding initialization...")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Process embeddings in batches
        await batch_process_embeddings(db, batch_size=50)
        
        # Get final stats
        stats = vector_store.get_stats()
        print(f"\nInitialization complete!")
        print(f"Total embeddings in index: {stats['total_embeddings']}")
        print(f"Memory usage: {stats['memory_usage_mb']:.2f} MB")
        print(f"Content type distribution: {stats['content_types']}")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())