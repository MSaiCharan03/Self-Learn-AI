from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from ..core.database import get_db
from ..models.user import User
from ..models.conversation import VectorEmbedding, KnowledgeBase
from ..utils.security import get_current_active_user, sanitize_input
from ..utils.helpers import semantic_search, find_similar_messages, find_relevant_knowledge, create_embedding_for_knowledge, get_embedding_stats

router = APIRouter(prefix="/search", tags=["search"])

class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    content_types: Optional[List[str]] = None  # ['message', 'response', 'knowledge']
    min_similarity: float = 0.7

class SearchResult(BaseModel):
    id: str
    content_text: str
    content_type: str
    similarity_score: float
    created_at: datetime
    metadata: dict

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: int

class KnowledgeCreate(BaseModel):
    title: str
    content: str
    source: Optional[str] = None
    content_type: str = "text"
    metadata: dict = {}

class KnowledgeResponse(BaseModel):
    id: str
    title: str
    content: str
    source: Optional[str]
    content_type: str
    created_at: datetime
    is_active: bool
    metadata: dict

@router.post("/semantic", response_model=SearchResponse)
async def semantic_search_endpoint(
    search_query: SearchQuery,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Perform semantic search using vector embeddings"""
    # Sanitize query
    query = sanitize_input(search_query.query)
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty"
        )
    
    start_time = datetime.now()
    
    # Perform actual semantic search
    search_results = await semantic_search(
        query=query,
        top_k=search_query.limit,
        min_similarity=search_query.min_similarity,
        content_types=search_query.content_types
    )
    
    # Get full content from database
    results = []
    for result in search_results:
        embedding = db.query(VectorEmbedding).filter(
            VectorEmbedding.embedding_id == result["embedding_id"]
        ).first()
        
        if embedding:
            results.append(SearchResult(
                id=str(embedding.id),
                content_text=embedding.content_text,
                content_type=embedding.content_type,
                similarity_score=result["similarity_score"],
                created_at=embedding.created_at,
                metadata=embedding.metadata
            ))
    
    search_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    return SearchResponse(
        query=query,
        results=results,
        total_results=len(results),
        search_time_ms=search_time
    )

@router.get("/knowledge", response_model=List[KnowledgeResponse])
async def get_knowledge_base(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
    content_type: Optional[str] = None,
    is_active: bool = True
):
    """Get knowledge base entries"""
    query = db.query(KnowledgeBase).filter(
        KnowledgeBase.is_active == is_active
    )
    
    if content_type:
        query = query.filter(KnowledgeBase.content_type == content_type)
    
    knowledge_entries = query.order_by(
        KnowledgeBase.created_at.desc()
    ).offset(offset).limit(limit).all()
    
    return [
        KnowledgeResponse(
            id=str(entry.id),
            title=entry.title,
            content=entry.content,
            source=entry.source,
            content_type=entry.content_type,
            created_at=entry.created_at,
            is_active=entry.is_active,
            metadata=entry.metadata
        ) for entry in knowledge_entries
    ]

@router.post("/knowledge", response_model=KnowledgeResponse)
async def create_knowledge_entry(
    knowledge_data: KnowledgeCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new knowledge base entry"""
    # Sanitize inputs
    title = sanitize_input(knowledge_data.title)
    content = sanitize_input(knowledge_data.content, max_length=50000)
    source = sanitize_input(knowledge_data.source) if knowledge_data.source else None
    
    if not title or not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title and content are required"
        )
    
    # Create knowledge entry
    knowledge_entry = KnowledgeBase(
        title=title,
        content=content,
        source=source,
        content_type=knowledge_data.content_type,
        metadata=knowledge_data.metadata
    )
    
    db.add(knowledge_entry)
    db.commit()
    db.refresh(knowledge_entry)
    
    # Generate and store embedding for the knowledge entry
    await create_embedding_for_knowledge(db, knowledge_entry)
    
    return KnowledgeResponse(
        id=str(knowledge_entry.id),
        title=knowledge_entry.title,
        content=knowledge_entry.content,
        source=knowledge_entry.source,
        content_type=knowledge_entry.content_type,
        created_at=knowledge_entry.created_at,
        is_active=knowledge_entry.is_active,
        metadata=knowledge_entry.metadata
    )

@router.get("/knowledge/{knowledge_id}", response_model=KnowledgeResponse)
async def get_knowledge_entry(
    knowledge_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific knowledge base entry"""
    knowledge_entry = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == knowledge_id
    ).first()
    
    if not knowledge_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge entry not found"
        )
    
    return KnowledgeResponse(
        id=str(knowledge_entry.id),
        title=knowledge_entry.title,
        content=knowledge_entry.content,
        source=knowledge_entry.source,
        content_type=knowledge_entry.content_type,
        created_at=knowledge_entry.created_at,
        is_active=knowledge_entry.is_active,
        metadata=knowledge_entry.metadata
    )

@router.put("/knowledge/{knowledge_id}", response_model=KnowledgeResponse)
async def update_knowledge_entry(
    knowledge_id: str,
    knowledge_data: KnowledgeCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update a knowledge base entry"""
    knowledge_entry = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == knowledge_id
    ).first()
    
    if not knowledge_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge entry not found"
        )
    
    # Sanitize inputs
    title = sanitize_input(knowledge_data.title)
    content = sanitize_input(knowledge_data.content, max_length=50000)
    source = sanitize_input(knowledge_data.source) if knowledge_data.source else None
    
    if not title or not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title and content are required"
        )
    
    # Update knowledge entry
    knowledge_entry.title = title
    knowledge_entry.content = content
    knowledge_entry.source = source
    knowledge_entry.content_type = knowledge_data.content_type
    knowledge_entry.metadata = knowledge_data.metadata
    knowledge_entry.updated_at = datetime.utcnow()
    
    db.commit()
    
    # Update embedding for the knowledge entry
    await create_embedding_for_knowledge(db, knowledge_entry)
    
    return KnowledgeResponse(
        id=str(knowledge_entry.id),
        title=knowledge_entry.title,
        content=knowledge_entry.content,
        source=knowledge_entry.source,
        content_type=knowledge_entry.content_type,
        created_at=knowledge_entry.created_at,
        is_active=knowledge_entry.is_active,
        metadata=knowledge_entry.metadata
    )

@router.delete("/knowledge/{knowledge_id}")
async def delete_knowledge_entry(
    knowledge_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a knowledge base entry"""
    knowledge_entry = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == knowledge_id
    ).first()
    
    if not knowledge_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge entry not found"
        )
    
    # Soft delete by setting is_active to False
    knowledge_entry.is_active = False
    knowledge_entry.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Knowledge entry deleted successfully"}

@router.get("/similar-messages/{message_id}")
async def get_similar_messages(
    message_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    top_k: int = 5,
    min_similarity: float = 0.8
):
    """Get similar messages for a given message"""
    # Get the message
    from ..models.conversation import Message
    message = db.query(Message).filter(Message.id == message_id).first()
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    # Find similar messages
    similar_messages = await find_similar_messages(
        db, message.content, top_k, min_similarity
    )
    
    return {
        "message_id": message_id,
        "query_content": message.content,
        "similar_messages": similar_messages
    }

@router.get("/relevant-knowledge")
async def get_relevant_knowledge(
    query: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    top_k: int = 3,
    min_similarity: float = 0.7
):
    """Get relevant knowledge base entries for a query"""
    # Sanitize query
    query = sanitize_input(query)
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    # Find relevant knowledge
    relevant_knowledge = await find_relevant_knowledge(
        db, query, top_k, min_similarity
    )
    
    return {
        "query": query,
        "relevant_knowledge": relevant_knowledge
    }

@router.get("/stats")
async def get_search_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get embedding and search statistics"""
    stats = get_embedding_stats(db)
    return stats