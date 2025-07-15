from sqlalchemy import Column, String, Boolean, DateTime, UUID, ForeignKey, Text, Integer, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from uuid import uuid4

Base = declarative_base()

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    response_id = Column(UUID(as_uuid=True), ForeignKey("model_responses.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    rating = Column(Integer)  # 1-5 scale
    thumbs_up = Column(Boolean)
    comment = Column(Text)
    feedback_type = Column(String(50), default='rating')  # 'rating', 'correction', 'preference'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata = Column(JSON, default=dict)
    
    # Relationships
    response = relationship("ModelResponse", back_populates="feedback")
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, rating={self.rating}, thumbs_up={self.thumbs_up})>"

class TrainingSession(Base):
    __tablename__ = "training_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name = Column(String(100), nullable=False)
    training_type = Column(String(50), nullable=False)  # 'fine_tune', 'lora', 'feedback_update'
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    status = Column(String(20), default='running')  # 'running', 'completed', 'failed'
    data_size = Column(Integer)
    loss_metrics = Column(JSON, default=dict)
    model_checkpoint_path = Column(String(500))
    parameters = Column(JSON, default=dict)
    error_log = Column(Text)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    def __repr__(self):
        return f"<TrainingSession(id={self.id}, model={self.model_name}, status={self.status})>"

class ModelComparison(Base):
    __tablename__ = "model_comparisons"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    prompt_text = Column(Text, nullable=False)
    phi2_response_id = Column(UUID(as_uuid=True), ForeignKey("model_responses.id"))
    external_response_id = Column(UUID(as_uuid=True), ForeignKey("model_responses.id"))
    winner = Column(String(50))  # 'phi2', 'external', 'tie'
    comparison_criteria = Column(JSON, default=dict)
    human_preference = Column(String(50))  # From user feedback
    automated_score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<ModelComparison(id={self.id}, winner={self.winner})>"

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id})>"