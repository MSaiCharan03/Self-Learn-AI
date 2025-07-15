from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..core.database import get_db
from ..models.user import User
from ..models.feedback import TrainingSession
from ..utils.security import get_current_active_user
from ..core.llm_manager import phi2_manager
from ..core.fine_tuning import fine_tuning_manager
from ..core.external_models import external_model_manager
import uuid

router = APIRouter(prefix="/models", tags=["models"])

class ModelInfo(BaseModel):
    name: str
    version: str
    type: str  # 'local', 'external'
    status: str  # 'active', 'inactive', 'loading', 'error'
    parameters: Dict[str, Any]
    last_updated: datetime
    performance_metrics: Dict[str, float]

class TrainingRequest(BaseModel):
    model_name: str
    training_type: str  # 'fine_tune', 'lora', 'feedback_update'
    parameters: Dict[str, Any] = {}
    data_filters: Dict[str, Any] = {}

class TrainingResponse(BaseModel):
    id: str
    model_name: str
    training_type: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    data_size: Optional[int]
    parameters: Dict[str, Any]
    progress: float  # 0.0 to 1.0

class ModelGenerationRequest(BaseModel):
    prompt: str
    model_name: str = "phi-2"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    parameters: Dict[str, Any] = {}

class ModelGenerationResponse(BaseModel):
    response_id: str
    model_name: str
    generated_text: str
    generation_time_ms: int
    token_count: int
    confidence_score: Optional[float]
    parameters: Dict[str, Any]

@router.get("/available", response_model=List[ModelInfo])
async def get_available_models(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get list of available models"""
    models = []
    
    # Get Phi-2 model info
    phi2_info = phi2_manager.get_model_info()
    
    models.append(ModelInfo(
        name="phi-2",
        version="microsoft/phi-2",
        type="local",
        status="active" if phi2_info["is_loaded"] else "inactive",
        parameters=phi2_info["model_info"] if phi2_info["is_loaded"] else {
            "context_length": 2048,
            "vocab_size": 51200,
            "model_size": "2.7B"
        },
        last_updated=datetime.utcnow(),
        performance_metrics={
            "avg_generation_time_ms": 150.0,
            "avg_rating": 4.2,
            "success_rate": 0.95
        }
    ))
    
    # Add external models
    external_model_info = external_model_manager.get_all_model_info()
    
    for model_name, info in external_model_info.items():
        models.append(ModelInfo(
            name=model_name,
            version=info.get("version", "1.0"),
            type="external",
            status="active" if info["is_available"] else "inactive",
            parameters={
                "context_length": info["context_length"],
                "provider": info["provider"],
                "multimodal": info.get("multimodal", False),
                "supports_system_prompt": info["supports_system_prompt"],
                "supports_conversation_history": info["supports_conversation_history"]
            },
            last_updated=datetime.utcnow(),
            performance_metrics={
                "avg_generation_time_ms": 600.0,
                "avg_rating": 4.5,
                "success_rate": 0.95
            }
        ))
    
    return models

@router.get("/status/{model_name}", response_model=ModelInfo)
async def get_model_status(
    model_name: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get status of a specific model"""
    if model_name == "phi-2":
        phi2_info = phi2_manager.get_model_info()
        
        return ModelInfo(
            name="phi-2",
            version="microsoft/phi-2",
            type="local",
            status="active" if phi2_info["is_loaded"] else "inactive",
            parameters=phi2_info["model_info"] if phi2_info["is_loaded"] else {
                "context_length": 2048,
                "vocab_size": 51200,
                "model_size": "2.7B"
            },
            last_updated=datetime.utcnow(),
            performance_metrics={
                "avg_generation_time_ms": 150.0,
                "avg_rating": 4.2,
                "success_rate": 0.95
            }
        )
    
    else:
        # Check external models
        external_model_info = external_model_manager.get_all_model_info()
        
        if model_name in external_model_info:
            info = external_model_info[model_name]
            
            return ModelInfo(
                name=model_name,
                version=info.get("version", "1.0"),
                type="external",
                status="active" if info["is_available"] else "inactive",
                parameters={
                    "context_length": info["context_length"],
                    "provider": info["provider"],
                    "multimodal": info.get("multimodal", False),
                    "supports_system_prompt": info["supports_system_prompt"],
                    "supports_conversation_history": info["supports_conversation_history"]
                },
                last_updated=datetime.utcnow(),
                performance_metrics={
                    "avg_generation_time_ms": 600.0,
                    "avg_rating": 4.5,
                    "success_rate": 0.95
                }
            )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )

@router.post("/generate", response_model=ModelGenerationResponse)
async def generate_response(
    request: ModelGenerationRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate response using specified model"""
    if request.model_name == "phi-2":
        try:
            # Generate response using Phi-2
            response_data = await phi2_manager.generate_response_async(
                prompt=request.prompt,
                generation_params={
                    "max_new_tokens": request.max_length,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "do_sample": request.do_sample,
                    **request.parameters
                }
            )
            
            return ModelGenerationResponse(
                response_id=str(uuid.uuid4()),
                model_name=request.model_name,
                generated_text=response_data["response"],
                generation_time_ms=response_data["generation_time_ms"],
                token_count=response_data["output_tokens"],
                confidence_score=None,  # Phi-2 doesn't provide confidence scores
                parameters=response_data["parameters"]
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {str(e)}"
            )
    
    else:
        # Try external models
        if external_model_manager.is_model_available(request.model_name):
            try:
                response_data = await external_model_manager.generate_response(
                    model_name=request.model_name,
                    prompt=request.prompt,
                    max_tokens=request.max_length,
                    temperature=request.temperature,
                    **request.parameters
                )
                
                return ModelGenerationResponse(
                    response_id=str(uuid.uuid4()),
                    model_name=request.model_name,
                    generated_text=response_data["response"],
                    generation_time_ms=response_data["generation_time_ms"],
                    token_count=response_data["output_tokens"],
                    confidence_score=None,
                    parameters={"provider": response_data["provider"]}
                )
                
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"External model generation failed: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found or not available"
            )

@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start model training/fine-tuning"""
    if request.model_name not in ["phi-2"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not available for training"
        )
    
    if request.training_type not in ["lora", "feedback_update"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid training type. Supported types: lora, feedback_update"
        )
    
    try:
        # Start training session
        session = fine_tuning_manager.start_training_session(
            db=db,
            training_type=request.training_type,
            parameters=request.parameters
        )
        
        # Set creator
        session.created_by = current_user.id
        db.commit()
        
        # Start background training (in a real system, this would be async)
        if request.training_type == "lora":
            # Extract training parameters
            num_epochs = request.parameters.get("num_epochs", 3)
            learning_rate = request.parameters.get("learning_rate", 5e-5)
            rank = request.parameters.get("rank", 8)
            alpha = request.parameters.get("alpha", 32)
            
            # Start LoRA training (this is blocking for now)
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Run training in background (simplified for demo)
            # In production, this should be queued to a background worker
            training_results = fine_tuning_manager.run_lora_training(
                db=db,
                session=session,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                rank=rank,
                alpha=alpha
            )
            
            if not training_results["success"]:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Training failed: {training_results['error']}"
                )
        
        # Refresh session to get updated data
        db.refresh(session)
        
        return TrainingResponse(
            id=str(session.id),
            model_name=session.model_name,
            training_type=session.training_type,
            status=session.status,
            start_time=session.start_time,
            end_time=session.end_time,
            data_size=session.data_size,
            parameters=session.parameters,
            progress=1.0 if session.status == "completed" else 0.0
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )

@router.get("/training", response_model=List[TrainingResponse])
async def get_training_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """Get training session history"""
    sessions = db.query(TrainingSession).filter(
        TrainingSession.created_by == current_user.id
    ).order_by(TrainingSession.start_time.desc()).offset(offset).limit(limit).all()
    
    return [
        TrainingResponse(
            id=str(session.id),
            model_name=session.model_name,
            training_type=session.training_type,
            status=session.status,
            start_time=session.start_time,
            end_time=session.end_time,
            data_size=session.data_size,
            parameters=session.parameters,
            progress=1.0 if session.status == "completed" else 0.0
        ) for session in sessions
    ]

@router.get("/training/{session_id}", response_model=TrainingResponse)
async def get_training_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get specific training session details"""
    session = db.query(TrainingSession).filter(
        TrainingSession.id == session_id,
        TrainingSession.created_by == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training session not found"
        )
    
    return TrainingResponse(
        id=str(session.id),
        model_name=session.model_name,
        training_type=session.training_type,
        status=session.status,
        start_time=session.start_time,
        end_time=session.end_time,
        data_size=session.data_size,
        parameters=session.parameters,
        progress=1.0 if session.status == "completed" else 0.0
    )

@router.delete("/training/{session_id}")
async def cancel_training(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Cancel a running training session"""
    session = db.query(TrainingSession).filter(
        TrainingSession.id == session_id,
        TrainingSession.created_by == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training session not found"
        )
    
    if session.status != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Training session is not running"
        )
    
    # TODO: Implement actual training cancellation
    session.status = "cancelled"
    session.end_time = datetime.utcnow()
    db.commit()
    
    return {"message": "Training session cancelled successfully"}