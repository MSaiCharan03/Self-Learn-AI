from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..core.database import get_db
from ..models.user import User
from ..models.feedback import TrainingSession, Feedback
from ..models.conversation import Message, ModelResponse
from ..utils.security import get_current_active_user
from ..core.fine_tuning import fine_tuning_manager

router = APIRouter(prefix="/training", tags=["training"])

class TrainingDataStats(BaseModel):
    total_conversations: int
    total_messages: int
    total_feedback: int
    average_rating: float
    rating_distribution: Dict[int, int]
    recent_feedback_sample: List[Dict[str, Any]]

class TrainingSchedule(BaseModel):
    enabled: bool
    schedule_type: str  # 'daily', 'weekly', 'on_feedback_threshold'
    feedback_threshold: int
    min_examples: int
    parameters: Dict[str, Any]

@router.get("/data-stats", response_model=TrainingDataStats)
async def get_training_data_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get statistics about available training data"""
    try:
        # Basic counts
        total_conversations = db.query(Message.conversation_id).distinct().count()
        total_messages = db.query(Message).count()
        total_feedback = db.query(Feedback).count()
        
        # Average rating
        avg_rating = db.query(Feedback).with_entities(
            db.func.avg(Feedback.rating)
        ).scalar() or 0.0
        
        # Rating distribution
        rating_distribution = {}
        for rating in range(1, 6):
            count = db.query(Feedback).filter(Feedback.rating == rating).count()
            rating_distribution[rating] = count
        
        # Recent feedback samples
        recent_feedback = db.query(Feedback).order_by(
            Feedback.created_at.desc()
        ).limit(5).all()
        
        feedback_sample = []
        for feedback in recent_feedback:
            # Get the associated response
            response = db.query(ModelResponse).filter(
                ModelResponse.id == feedback.response_id
            ).first()
            
            if response:
                # Get the associated message
                message = db.query(Message).filter(
                    Message.id == response.message_id
                ).first()
                
                feedback_sample.append({
                    "rating": feedback.rating,
                    "thumbs_up": feedback.thumbs_up,
                    "comment": feedback.comment,
                    "created_at": feedback.created_at.isoformat(),
                    "prompt": message.content[:100] + "..." if message and len(message.content) > 100 else message.content if message else "N/A",
                    "response": response.response_text[:100] + "..." if len(response.response_text) > 100 else response.response_text
                })
        
        return TrainingDataStats(
            total_conversations=total_conversations,
            total_messages=total_messages,
            total_feedback=total_feedback,
            average_rating=float(avg_rating),
            rating_distribution=rating_distribution,
            recent_feedback_sample=feedback_sample
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training data stats: {str(e)}"
        )

@router.post("/prepare-data")
async def prepare_training_data(
    min_examples: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Prepare and validate training data"""
    try:
        # Prepare training data
        train_dataset, stats = fine_tuning_manager.prepare_training_data(
            db=db,
            min_examples=min_examples
        )
        
        return {
            "success": True,
            "dataset_size": len(train_dataset),
            "stats": stats,
            "message": f"Training data prepared with {len(train_dataset)} examples"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prepare training data: {str(e)}"
        )

@router.post("/start-lora")
async def start_lora_training(
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    rank: int = 8,
    alpha: int = 32,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start LoRA fine-tuning"""
    try:
        # Start training session
        session = fine_tuning_manager.start_training_session(
            db=db,
            training_type="lora",
            parameters={
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "rank": rank,
                "alpha": alpha
            }
        )
        
        # Set creator
        session.created_by = current_user.id
        db.commit()
        
        # Start LoRA training
        training_results = fine_tuning_manager.run_lora_training(
            db=db,
            session=session,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            rank=rank,
            alpha=alpha
        )
        
        if training_results["success"]:
            return {
                "success": True,
                "session_id": training_results["session_id"],
                "output_dir": training_results["output_dir"],
                "training_results": training_results["training_results"],
                "data_stats": training_results["data_stats"]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Training failed: {training_results['error']}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )

@router.get("/sessions", response_model=List[Dict[str, Any]])
async def get_training_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 20,
    offset: int = 0
):
    """Get training session history"""
    try:
        sessions = db.query(TrainingSession).order_by(
            TrainingSession.start_time.desc()
        ).offset(offset).limit(limit).all()
        
        session_data = []
        for session in sessions:
            # Get creator info
            creator = db.query(User).filter(User.id == session.created_by).first()
            
            session_data.append({
                "id": str(session.id),
                "model_name": session.model_name,
                "training_type": session.training_type,
                "status": session.status,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "data_size": session.data_size,
                "parameters": session.parameters,
                "loss_metrics": session.loss_metrics,
                "model_checkpoint_path": session.model_checkpoint_path,
                "error_log": session.error_log,
                "created_by": creator.username if creator else "Unknown",
                "duration": str(session.end_time - session.start_time) if session.end_time else None
            })
        
        return session_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training sessions: {str(e)}"
        )

@router.get("/sessions/{session_id}")
async def get_training_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get specific training session details"""
    try:
        session = db.query(TrainingSession).filter(
            TrainingSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training session not found"
            )
        
        # Get creator info
        creator = db.query(User).filter(User.id == session.created_by).first()
        
        return {
            "id": str(session.id),
            "model_name": session.model_name,
            "training_type": session.training_type,
            "status": session.status,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "data_size": session.data_size,
            "parameters": session.parameters,
            "loss_metrics": session.loss_metrics,
            "model_checkpoint_path": session.model_checkpoint_path,
            "error_log": session.error_log,
            "created_by": creator.username if creator else "Unknown",
            "duration": str(session.end_time - session.start_time) if session.end_time else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training session: {str(e)}"
        )

@router.post("/evaluate/{session_id}")
async def evaluate_training_session(
    session_id: str,
    test_prompts: Optional[List[str]] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Evaluate a training session"""
    try:
        session = db.query(TrainingSession).filter(
            TrainingSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training session not found"
            )
        
        if session.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training session is not completed"
            )
        
        # Run evaluation
        evaluation_results = fine_tuning_manager.evaluate_model_improvement(
            db=db,
            checkpoint_path=session.model_checkpoint_path,
            test_prompts=test_prompts
        )
        
        return {
            "session_id": session_id,
            "evaluation_results": evaluation_results,
            "checkpoint_path": session.model_checkpoint_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate training session: {str(e)}"
        )

@router.get("/quality-metrics")
async def get_quality_metrics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    days: int = 30
):
    """Get model quality metrics over time"""
    try:
        from datetime import datetime, timedelta
        
        # Get feedback from last N days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        feedback_query = db.query(Feedback).filter(
            Feedback.created_at >= cutoff_date
        ).order_by(Feedback.created_at.asc())
        
        feedback_data = []
        for feedback in feedback_query:
            feedback_data.append({
                "rating": feedback.rating,
                "thumbs_up": feedback.thumbs_up,
                "created_at": feedback.created_at.isoformat(),
                "feedback_type": feedback.feedback_type
            })
        
        # Calculate metrics
        total_feedback = len(feedback_data)
        if total_feedback == 0:
            return {
                "total_feedback": 0,
                "average_rating": 0.0,
                "thumbs_up_rate": 0.0,
                "feedback_trend": [],
                "improvement_suggestions": []
            }
        
        avg_rating = sum(f["rating"] for f in feedback_data if f["rating"]) / total_feedback
        thumbs_up_count = sum(1 for f in feedback_data if f["thumbs_up"] is True)
        thumbs_up_rate = thumbs_up_count / total_feedback
        
        # Get improvement suggestions
        low_rating_count = sum(1 for f in feedback_data if f["rating"] and f["rating"] <= 2)
        suggestions = []
        
        if low_rating_count > total_feedback * 0.3:
            suggestions.append("Consider running fine-tuning to improve response quality")
        
        if thumbs_up_rate < 0.6:
            suggestions.append("Response quality is below average - review recent feedback")
        
        if total_feedback < 10:
            suggestions.append("Collect more user feedback to improve training data")
        
        return {
            "total_feedback": total_feedback,
            "average_rating": avg_rating,
            "thumbs_up_rate": thumbs_up_rate,
            "low_rating_percentage": low_rating_count / total_feedback,
            "feedback_trend": feedback_data,
            "improvement_suggestions": suggestions
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quality metrics: {str(e)}"
        )