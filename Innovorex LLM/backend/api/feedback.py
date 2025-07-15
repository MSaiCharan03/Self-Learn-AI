from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from ..core.database import get_db
from ..models.user import User
from ..models.conversation import ModelResponse
from ..models.feedback import Feedback, ModelComparison
from ..utils.security import get_current_active_user, sanitize_input

router = APIRouter(prefix="/feedback", tags=["feedback"])

class FeedbackCreate(BaseModel):
    response_id: str
    rating: Optional[int] = None  # 1-5 scale
    thumbs_up: Optional[bool] = None
    comment: Optional[str] = None
    feedback_type: str = "rating"

class FeedbackResponse(BaseModel):
    id: str
    response_id: str
    rating: Optional[int]
    thumbs_up: Optional[bool]
    comment: Optional[str]
    feedback_type: str
    created_at: datetime

class ModelComparisonCreate(BaseModel):
    prompt_text: str
    phi2_response_id: str
    external_response_id: str
    winner: str  # 'phi2', 'external', 'tie'
    human_preference: Optional[str] = None

class ModelComparisonResponse(BaseModel):
    id: str
    prompt_text: str
    winner: str
    human_preference: Optional[str]
    automated_score: Optional[float]
    created_at: datetime

class ModelStats(BaseModel):
    model_name: str
    total_responses: int
    avg_rating: Optional[float]
    thumbs_up_count: int
    thumbs_down_count: int
    avg_generation_time_ms: Optional[float]

@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback_data: FeedbackCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit feedback for a model response"""
    # Validate response exists
    response = db.query(ModelResponse).filter(
        ModelResponse.id == feedback_data.response_id
    ).first()
    
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model response not found"
        )
    
    # Validate feedback data
    if feedback_data.rating is not None and (feedback_data.rating < 1 or feedback_data.rating > 5):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be between 1 and 5"
        )
    
    if feedback_data.rating is None and feedback_data.thumbs_up is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either rating or thumbs_up must be provided"
        )
    
    # Sanitize comment
    comment = sanitize_input(feedback_data.comment) if feedback_data.comment else None
    
    # Check if feedback already exists for this response and user
    existing_feedback = db.query(Feedback).filter(
        Feedback.response_id == feedback_data.response_id,
        Feedback.user_id == current_user.id
    ).first()
    
    if existing_feedback:
        # Update existing feedback
        existing_feedback.rating = feedback_data.rating
        existing_feedback.thumbs_up = feedback_data.thumbs_up
        existing_feedback.comment = comment
        existing_feedback.feedback_type = feedback_data.feedback_type
        existing_feedback.updated_at = datetime.utcnow()
        db.commit()
        
        feedback = existing_feedback
    else:
        # Create new feedback
        feedback = Feedback(
            response_id=feedback_data.response_id,
            user_id=current_user.id,
            rating=feedback_data.rating,
            thumbs_up=feedback_data.thumbs_up,
            comment=comment,
            feedback_type=feedback_data.feedback_type
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
    
    return FeedbackResponse(
        id=str(feedback.id),
        response_id=str(feedback.response_id),
        rating=feedback.rating,
        thumbs_up=feedback.thumbs_up,
        comment=feedback.comment,
        feedback_type=feedback.feedback_type,
        created_at=feedback.created_at
    )

@router.get("/response/{response_id}", response_model=List[FeedbackResponse])
async def get_response_feedback(
    response_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all feedback for a specific response"""
    # Verify response exists
    response = db.query(ModelResponse).filter(
        ModelResponse.id == response_id
    ).first()
    
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model response not found"
        )
    
    feedbacks = db.query(Feedback).filter(
        Feedback.response_id == response_id
    ).order_by(Feedback.created_at.desc()).all()
    
    return [
        FeedbackResponse(
            id=str(f.id),
            response_id=str(f.response_id),
            rating=f.rating,
            thumbs_up=f.thumbs_up,
            comment=f.comment,
            feedback_type=f.feedback_type,
            created_at=f.created_at
        ) for f in feedbacks
    ]

@router.get("/my-feedback", response_model=List[FeedbackResponse])
async def get_my_feedback(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """Get current user's feedback history"""
    feedbacks = db.query(Feedback).filter(
        Feedback.user_id == current_user.id
    ).order_by(Feedback.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        FeedbackResponse(
            id=str(f.id),
            response_id=str(f.response_id),
            rating=f.rating,
            thumbs_up=f.thumbs_up,
            comment=f.comment,
            feedback_type=f.feedback_type,
            created_at=f.created_at
        ) for f in feedbacks
    ]

@router.post("/compare", response_model=ModelComparisonResponse)
async def submit_model_comparison(
    comparison_data: ModelComparisonCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit a model comparison result"""
    # Validate responses exist
    phi2_response = db.query(ModelResponse).filter(
        ModelResponse.id == comparison_data.phi2_response_id
    ).first()
    
    external_response = db.query(ModelResponse).filter(
        ModelResponse.id == comparison_data.external_response_id
    ).first()
    
    if not phi2_response or not external_response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or both model responses not found"
        )
    
    # Validate winner
    if comparison_data.winner not in ['phi2', 'external', 'tie']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Winner must be 'phi2', 'external', or 'tie'"
        )
    
    # Create comparison record
    comparison = ModelComparison(
        prompt_text=sanitize_input(comparison_data.prompt_text),
        phi2_response_id=comparison_data.phi2_response_id,
        external_response_id=comparison_data.external_response_id,
        winner=comparison_data.winner,
        human_preference=comparison_data.human_preference
    )
    
    db.add(comparison)
    db.commit()
    db.refresh(comparison)
    
    return ModelComparisonResponse(
        id=str(comparison.id),
        prompt_text=comparison.prompt_text,
        winner=comparison.winner,
        human_preference=comparison.human_preference,
        automated_score=comparison.automated_score,
        created_at=comparison.created_at
    )

@router.get("/stats", response_model=List[ModelStats])
async def get_model_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get model performance statistics"""
    # Raw SQL query for better performance
    query = """
    SELECT 
        mr.model_name,
        COUNT(*) as total_responses,
        AVG(f.rating) as avg_rating,
        COUNT(CASE WHEN f.thumbs_up = true THEN 1 END) as thumbs_up_count,
        COUNT(CASE WHEN f.thumbs_up = false THEN 1 END) as thumbs_down_count,
        AVG(mr.generation_time_ms) as avg_generation_time_ms
    FROM model_responses mr
    LEFT JOIN feedback f ON mr.id = f.response_id
    GROUP BY mr.model_name
    ORDER BY total_responses DESC
    """
    
    result = db.execute(query).fetchall()
    
    return [
        ModelStats(
            model_name=row[0],
            total_responses=row[1],
            avg_rating=float(row[2]) if row[2] else None,
            thumbs_up_count=row[3],
            thumbs_down_count=row[4],
            avg_generation_time_ms=float(row[5]) if row[5] else None
        ) for row in result
    ]

@router.delete("/feedback/{feedback_id}")
async def delete_feedback(
    feedback_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete user's own feedback"""
    feedback = db.query(Feedback).filter(
        Feedback.id == feedback_id,
        Feedback.user_id == current_user.id
    ).first()
    
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found"
        )
    
    db.delete(feedback)
    db.commit()
    
    return {"message": "Feedback deleted successfully"}