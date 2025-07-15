from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import time
from ..core.database import get_db
from ..models.user import User
from ..models.conversation import Conversation, Message, ModelResponse
from ..utils.security import get_current_active_user, sanitize_input
from ..utils.helpers import create_embedding_for_message, create_embedding_for_response, find_similar_messages, find_relevant_knowledge
from ..core.llm_manager import phi2_manager
from ..core.external_models import external_model_manager
import uuid

router = APIRouter(prefix="/chat", tags=["chat"])

class MessageCreate(BaseModel):
    content: str
    conversation_id: Optional[str] = None
    generate_comparisons: bool = False  # Whether to generate responses from external models
    comparison_models: Optional[List[str]] = None  # Specific models to compare

class MessageResponse(BaseModel):
    id: str
    content: str
    message_type: str
    created_at: datetime
    token_count: int

class ModelResponseData(BaseModel):
    id: str
    model_name: str
    response_text: str
    confidence_score: Optional[float]
    generation_time_ms: Optional[int]
    token_count: int
    is_primary: bool
    created_at: datetime

class ChatResponse(BaseModel):
    message_id: str
    user_message: MessageResponse
    model_responses: List[ModelResponseData]
    conversation_id: str

class ConversationResponse(BaseModel):
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_at: Optional[datetime]

class ConversationHistory(BaseModel):
    id: str
    title: Optional[str]
    messages: List[MessageResponse]
    created_at: datetime
    updated_at: datetime

@router.post("/send", response_model=ChatResponse)
async def send_message(
    message_data: MessageCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Send a chat message and get AI response"""
    # Sanitize input
    content = sanitize_input(message_data.content)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty"
        )
    
    # Get or create conversation
    conversation = None
    if message_data.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == message_data.conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    else:
        # Create new conversation
        conversation = Conversation(
            user_id=current_user.id,
            title=content[:50] + "..." if len(content) > 50 else content
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    
    # Create user message
    user_message = Message(
        conversation_id=conversation.id,
        user_id=current_user.id,
        content=content,
        message_type="user",
        token_count=len(content.split())  # Simple word count
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)
    
    # Get conversation history for context
    conversation_history = []
    if conversation.id:
        recent_messages = db.query(Message).filter(
            Message.conversation_id == conversation.id
        ).order_by(Message.created_at.desc()).limit(10).all()
        
        # Format for LLM context
        for msg in reversed(recent_messages):
            conversation_history.append({
                "role": "user" if msg.message_type == "user" else "assistant",
                "content": msg.content
            })
    
    # Get RAG context
    rag_context = ""
    try:
        # Find similar messages
        similar_messages = await find_similar_messages(db, content, top_k=3, min_similarity=0.7)
        
        # Find relevant knowledge
        relevant_knowledge = await find_relevant_knowledge(db, content, top_k=2, min_similarity=0.7)
        
        # Build context
        context_parts = []
        
        if similar_messages:
            context_parts.append("Similar past conversations:")
            for msg in similar_messages[:2]:  # Limit to 2 for brevity
                context_parts.append(f"- {msg['content'][:100]}...")
        
        if relevant_knowledge:
            context_parts.append("Relevant knowledge:")
            for kb in relevant_knowledge[:2]:  # Limit to 2 for brevity
                context_parts.append(f"- {kb['title']}: {kb['content'][:150]}...")
        
        if context_parts:
            rag_context = "\n".join(context_parts) + "\n\n"
    
    except Exception as e:
        print(f"RAG context error: {e}")
        rag_context = ""
    
    # Generate AI response using Phi-2
    try:
        system_prompt = (
            "You are a helpful, harmless, and honest AI assistant. "
            "Provide clear, accurate, and helpful responses. "
            "Be concise but informative. "
            f"{rag_context}"
        )
        
        generation_start = time.time()
        response_data = await phi2_manager.generate_response_async(
            prompt=content,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            generation_params={
                "max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        )
        
        ai_response_text = response_data["response"]
        generation_time_ms = response_data["generation_time_ms"]
        token_count = response_data["output_tokens"]
        
    except Exception as e:
        print(f"LLM generation error: {e}")
        ai_response_text = "I apologize, but I'm having trouble generating a response right now. Please try again."
        generation_time_ms = 0
        token_count = 0
    
    # Create AI message
    ai_message = Message(
        conversation_id=conversation.id,
        user_id=current_user.id,
        content=ai_response_text,
        message_type="assistant",
        token_count=token_count
    )
    db.add(ai_message)
    db.commit()
    db.refresh(ai_message)
    
    # Create model response record for Phi-2
    model_response = ModelResponse(
        message_id=ai_message.id,
        model_name="phi-2",
        response_text=ai_response_text,
        generation_time_ms=generation_time_ms,
        token_count=token_count,
        is_primary=True
    )
    db.add(model_response)
    db.commit()
    db.refresh(model_response)
    
    # Store all model responses for comparison
    all_model_responses = [model_response]
    
    # Generate responses from external models if requested
    if message_data.generate_comparisons:
        available_models = external_model_manager.get_available_models()
        
        # Use specified models or default to available ones
        comparison_models = message_data.comparison_models or available_models
        
        # Filter to only available models
        models_to_use = [m for m in comparison_models if m in available_models]
        
        if models_to_use:
            try:
                # Generate responses from external models
                external_responses = await external_model_manager.compare_models(
                    model_names=models_to_use,
                    prompt=content,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    max_tokens=300,
                    temperature=0.7
                )
                
                # Store external model responses
                for model_name, response_data in external_responses.items():
                    if "error" not in response_data:
                        external_model_response = ModelResponse(
                            message_id=ai_message.id,
                            model_name=model_name,
                            response_text=response_data["response"],
                            generation_time_ms=response_data["generation_time_ms"],
                            token_count=response_data["output_tokens"],
                            is_primary=False,
                            parameters={"provider": response_data["provider"]}
                        )
                        db.add(external_model_response)
                        db.commit()
                        db.refresh(external_model_response)
                        all_model_responses.append(external_model_response)
            
            except Exception as e:
                print(f"Error generating external model responses: {e}")
    
    # Create embeddings for messages
    await create_embedding_for_message(db, user_message)
    await create_embedding_for_response(db, model_response)
    
    # Update conversation timestamp
    conversation.updated_at = datetime.utcnow()
    db.commit()
    
    return ChatResponse(
        message_id=str(ai_message.id),
        user_message=MessageResponse(
            id=str(user_message.id),
            content=user_message.content,
            message_type=user_message.message_type,
            created_at=user_message.created_at,
            token_count=user_message.token_count
        ),
        model_responses=[
            ModelResponseData(
                id=str(resp.id),
                model_name=resp.model_name,
                response_text=resp.response_text,
                confidence_score=resp.confidence_score,
                generation_time_ms=resp.generation_time_ms,
                token_count=resp.token_count,
                is_primary=resp.is_primary,
                created_at=resp.created_at
            ) for resp in all_model_responses
        ],
        conversation_id=str(conversation.id)
    )

@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """Get user's conversations"""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.is_archived == False
    ).order_by(Conversation.updated_at.desc()).offset(offset).limit(limit).all()
    
    result = []
    for conv in conversations:
        # Get message count and last message time
        messages = db.query(Message).filter(
            Message.conversation_id == conv.id
        ).order_by(Message.created_at.desc()).all()
        
        result.append(ConversationResponse(
            id=str(conv.id),
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=len(messages),
            last_message_at=messages[0].created_at if messages else None
        ))
    
    return result

@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get conversation history"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation.id
    ).order_by(Message.created_at.asc()).all()
    
    return ConversationHistory(
        id=str(conversation.id),
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageResponse(
                id=str(msg.id),
                content=msg.content,
                message_type=msg.message_type,
                created_at=msg.created_at,
                token_count=msg.token_count
            ) for msg in messages
        ]
    )

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    db.delete(conversation)
    db.commit()
    
    return {"message": "Conversation deleted successfully"}

@router.put("/conversations/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Archive a conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    conversation.is_archived = True
    db.commit()
    
    return {"message": "Conversation archived successfully"}