import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from ..models.conversation import VectorEmbedding, Message, ModelResponse, KnowledgeBase
from ..core.embeddings import embedding_manager
from ..core.vector_store import vector_store

async def create_embedding_for_message(
    db: Session,
    message: Message,
    content_type: str = "message"
) -> Optional[VectorEmbedding]:
    """Create and store embedding for a message"""
    try:
        # Generate embedding
        embedding_vector = embedding_manager.generate_embedding(message.content)
        embedding_id = embedding_manager.generate_embedding_id(message.content, content_type)
        
        # Create database record
        db_embedding = VectorEmbedding(
            embedding_id=embedding_id,
            content_type=content_type,
            content_id=message.id,
            content_text=message.content,
            model_name=embedding_manager.model_name,
            dimensions=embedding_manager.dimensions,
            metadata={
                "conversation_id": str(message.conversation_id),
                "user_id": str(message.user_id),
                "message_type": message.message_type,
                "token_count": message.token_count
            }
        )
        
        db.add(db_embedding)
        db.commit()
        db.refresh(db_embedding)
        
        # Add to Faiss index
        metadata = {
            "content_type": content_type,
            "content_id": str(message.id),
            "model_name": embedding_manager.model_name,
            "created_at": db_embedding.created_at.isoformat(),
            "metadata": db_embedding.metadata
        }
        
        vector_store.add_embedding(embedding_id, embedding_vector, metadata)
        
        # Update message with embedding_id
        message.embedding_id = embedding_id
        db.commit()
        
        return db_embedding
        
    except Exception as e:
        db.rollback()
        print(f"Error creating embedding for message {message.id}: {e}")
        return None

async def create_embedding_for_response(
    db: Session,
    response: ModelResponse,
    content_type: str = "response"
) -> Optional[VectorEmbedding]:
    """Create and store embedding for a model response"""
    try:
        # Generate embedding
        embedding_vector = embedding_manager.generate_embedding(response.response_text)
        embedding_id = embedding_manager.generate_embedding_id(response.response_text, content_type)
        
        # Create database record
        db_embedding = VectorEmbedding(
            embedding_id=embedding_id,
            content_type=content_type,
            content_id=response.id,
            content_text=response.response_text,
            model_name=embedding_manager.model_name,
            dimensions=embedding_manager.dimensions,
            metadata={
                "message_id": str(response.message_id),
                "model_name": response.model_name,
                "is_primary": response.is_primary,
                "generation_time_ms": response.generation_time_ms,
                "token_count": response.token_count
            }
        )
        
        db.add(db_embedding)
        db.commit()
        db.refresh(db_embedding)
        
        # Add to Faiss index
        metadata = {
            "content_type": content_type,
            "content_id": str(response.id),
            "model_name": embedding_manager.model_name,
            "created_at": db_embedding.created_at.isoformat(),
            "metadata": db_embedding.metadata
        }
        
        vector_store.add_embedding(embedding_id, embedding_vector, metadata)
        
        # Update response with embedding_id
        response.embedding_id = embedding_id
        db.commit()
        
        return db_embedding
        
    except Exception as e:
        db.rollback()
        print(f"Error creating embedding for response {response.id}: {e}")
        return None

async def create_embedding_for_knowledge(
    db: Session,
    knowledge: KnowledgeBase,
    content_type: str = "knowledge"
) -> Optional[VectorEmbedding]:
    """Create and store embedding for a knowledge base entry"""
    try:
        # Combine title and content for embedding
        combined_text = f"{knowledge.title}\n\n{knowledge.content}"
        
        # Generate embedding
        embedding_vector = embedding_manager.generate_embedding(combined_text)
        embedding_id = embedding_manager.generate_embedding_id(combined_text, content_type)
        
        # Create database record
        db_embedding = VectorEmbedding(
            embedding_id=embedding_id,
            content_type=content_type,
            content_id=knowledge.id,
            content_text=combined_text,
            model_name=embedding_manager.model_name,
            dimensions=embedding_manager.dimensions,
            metadata={
                "title": knowledge.title,
                "source": knowledge.source,
                "knowledge_content_type": knowledge.content_type,
                "is_active": knowledge.is_active
            }
        )
        
        db.add(db_embedding)
        db.commit()
        db.refresh(db_embedding)
        
        # Add to Faiss index
        metadata = {
            "content_type": content_type,
            "content_id": str(knowledge.id),
            "model_name": embedding_manager.model_name,
            "created_at": db_embedding.created_at.isoformat(),
            "metadata": db_embedding.metadata
        }
        
        vector_store.add_embedding(embedding_id, embedding_vector, metadata)
        
        # Update knowledge with embedding_id
        knowledge.embedding_id = embedding_id
        db.commit()
        
        return db_embedding
        
    except Exception as e:
        db.rollback()
        print(f"Error creating embedding for knowledge {knowledge.id}: {e}")
        return None

async def semantic_search(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.7,
    content_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Perform semantic search across all content types"""
    try:
        # Generate query embedding
        query_embedding = embedding_manager.generate_embedding(query)
        
        # Search in Faiss index
        results = vector_store.search(query_embedding, top_k, min_similarity)
        
        # Filter by content types if specified
        if content_types:
            results = [
                (emb_id, score, metadata) 
                for emb_id, score, metadata in results
                if metadata.get("content_type") in content_types
            ]
        
        # Format results
        formatted_results = []
        for embedding_id, similarity, metadata in results:
            formatted_results.append({
                "embedding_id": embedding_id,
                "similarity_score": similarity,
                "content_type": metadata.get("content_type"),
                "content_id": metadata.get("content_id"),
                "created_at": metadata.get("created_at"),
                "metadata": metadata.get("metadata", {})
            })
        
        return formatted_results
        
    except Exception as e:
        print(f"Error performing semantic search: {e}")
        return []

async def find_similar_messages(
    db: Session,
    message_content: str,
    top_k: int = 5,
    min_similarity: float = 0.8
) -> List[Dict[str, Any]]:
    """Find similar messages for RAG context"""
    try:
        # Search for similar messages
        results = await semantic_search(
            message_content,
            top_k=top_k,
            min_similarity=min_similarity,
            content_types=["message", "response"]
        )
        
        # Get full message/response data from database
        detailed_results = []
        for result in results:
            content_id = result["content_id"]
            content_type = result["content_type"]
            
            if content_type == "message":
                message = db.query(Message).filter(Message.id == content_id).first()
                if message:
                    detailed_results.append({
                        "type": "message",
                        "content": message.content,
                        "message_type": message.message_type,
                        "created_at": message.created_at.isoformat(),
                        "similarity": result["similarity_score"],
                        "conversation_id": str(message.conversation_id)
                    })
            
            elif content_type == "response":
                response = db.query(ModelResponse).filter(ModelResponse.id == content_id).first()
                if response:
                    detailed_results.append({
                        "type": "response",
                        "content": response.response_text,
                        "model_name": response.model_name,
                        "created_at": response.created_at.isoformat(),
                        "similarity": result["similarity_score"],
                        "generation_time_ms": response.generation_time_ms
                    })
        
        return detailed_results
        
    except Exception as e:
        print(f"Error finding similar messages: {e}")
        return []

async def find_relevant_knowledge(
    db: Session,
    query: str,
    top_k: int = 3,
    min_similarity: float = 0.7
) -> List[Dict[str, Any]]:
    """Find relevant knowledge base entries for RAG"""
    try:
        # Search for relevant knowledge
        results = await semantic_search(
            query,
            top_k=top_k,
            min_similarity=min_similarity,
            content_types=["knowledge"]
        )
        
        # Get full knowledge data from database
        detailed_results = []
        for result in results:
            content_id = result["content_id"]
            
            knowledge = db.query(KnowledgeBase).filter(
                KnowledgeBase.id == content_id,
                KnowledgeBase.is_active == True
            ).first()
            
            if knowledge:
                detailed_results.append({
                    "id": str(knowledge.id),
                    "title": knowledge.title,
                    "content": knowledge.content,
                    "source": knowledge.source,
                    "content_type": knowledge.content_type,
                    "similarity": result["similarity_score"],
                    "created_at": knowledge.created_at.isoformat()
                })
        
        return detailed_results
        
    except Exception as e:
        print(f"Error finding relevant knowledge: {e}")
        return []

async def batch_process_embeddings(
    db: Session,
    batch_size: int = 100,
    content_types: Optional[List[str]] = None
):
    """Process embeddings in batches for existing content"""
    try:
        print("Starting batch embedding processing...")
        
        # Process messages
        if not content_types or "message" in content_types:
            messages = db.query(Message).filter(Message.embedding_id.is_(None)).all()
            print(f"Processing {len(messages)} messages...")
            
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                for message in batch:
                    await create_embedding_for_message(db, message)
                    
                print(f"Processed {min(i + batch_size, len(messages))}/{len(messages)} messages")
                await asyncio.sleep(0.1)  # Small delay to prevent overload
        
        # Process responses
        if not content_types or "response" in content_types:
            responses = db.query(ModelResponse).filter(ModelResponse.embedding_id.is_(None)).all()
            print(f"Processing {len(responses)} responses...")
            
            for i in range(0, len(responses), batch_size):
                batch = responses[i:i + batch_size]
                for response in batch:
                    await create_embedding_for_response(db, response)
                    
                print(f"Processed {min(i + batch_size, len(responses))}/{len(responses)} responses")
                await asyncio.sleep(0.1)
        
        # Process knowledge base
        if not content_types or "knowledge" in content_types:
            knowledge_entries = db.query(KnowledgeBase).filter(
                KnowledgeBase.embedding_id.is_(None),
                KnowledgeBase.is_active == True
            ).all()
            print(f"Processing {len(knowledge_entries)} knowledge entries...")
            
            for i in range(0, len(knowledge_entries), batch_size):
                batch = knowledge_entries[i:i + batch_size]
                for knowledge in batch:
                    await create_embedding_for_knowledge(db, knowledge)
                    
                print(f"Processed {min(i + batch_size, len(knowledge_entries))}/{len(knowledge_entries)} knowledge entries")
                await asyncio.sleep(0.1)
        
        print("Batch embedding processing completed")
        
    except Exception as e:
        print(f"Error in batch embedding processing: {e}")
        raise

def get_embedding_stats(db: Session) -> Dict[str, Any]:
    """Get embedding statistics"""
    try:
        # Database stats
        total_embeddings = db.query(VectorEmbedding).count()
        message_embeddings = db.query(VectorEmbedding).filter(
            VectorEmbedding.content_type == "message"
        ).count()
        response_embeddings = db.query(VectorEmbedding).filter(
            VectorEmbedding.content_type == "response"
        ).count()
        knowledge_embeddings = db.query(VectorEmbedding).filter(
            VectorEmbedding.content_type == "knowledge"
        ).count()
        
        # Faiss stats
        faiss_stats = vector_store.get_stats()
        
        return {
            "database": {
                "total_embeddings": total_embeddings,
                "message_embeddings": message_embeddings,
                "response_embeddings": response_embeddings,
                "knowledge_embeddings": knowledge_embeddings
            },
            "faiss_index": faiss_stats,
            "embedding_model": embedding_manager.get_model_info()
        }
        
    except Exception as e:
        print(f"Error getting embedding stats: {e}")
        return {"error": str(e)}