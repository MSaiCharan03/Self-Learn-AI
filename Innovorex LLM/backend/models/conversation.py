from sqlalchemy import Column, String, Boolean, DateTime, UUID, ForeignKey, Text, Integer, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from uuid import uuid4

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_archived = Column(Boolean, default=False)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title})>"

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(String(20), nullable=False)  # 'user' or 'assistant'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    embedding_id = Column(String(255))
    token_count = Column(Integer, default=0)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    model_responses = relationship("ModelResponse", back_populates="message", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Message(id={self.id}, type={self.message_type})>"

class ModelResponse(Base):
    __tablename__ = "model_responses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String(100), nullable=False)
    response_text = Column(Text, nullable=False)
    confidence_score = Column(Float)
    generation_time_ms = Column(Integer)
    token_count = Column(Integer, default=0)
    embedding_id = Column(String(255))
    model_version = Column(String(50))
    parameters = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_primary = Column(Boolean, default=False)
    
    # Relationships
    message = relationship("Message", back_populates="model_responses")
    feedback = relationship("Feedback", back_populates="response", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ModelResponse(id={self.id}, model={self.model_name})>"

class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    embedding_id = Column(String(255), unique=True, nullable=False)
    content_type = Column(String(50), nullable=False)  # 'message', 'response', 'knowledge'
    content_id = Column(UUID(as_uuid=True), nullable=False)
    content_text = Column(Text, nullable=False)
    model_name = Column(String(100), nullable=False)
    dimensions = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<VectorEmbedding(id={self.embedding_id}, type={self.content_type})>"

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(255))
    content_type = Column(String(50), default='text')
    embedding_id = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<KnowledgeBase(id={self.id}, title={self.title})>"