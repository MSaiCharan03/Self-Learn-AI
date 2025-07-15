from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..core.database import get_db
from ..models.user import User
from ..models.conversation import ModelResponse
from ..models.feedback import ModelComparison
from ..utils.security import get_current_active_user, sanitize_input
from ..core.external_models import external_model_manager

router = APIRouter(prefix="/external", tags=["external_models"])

class ExternalModelInfo(BaseModel):
    model_name: str
    provider: str
    is_available: bool
    is_configured: bool
    context_length: int
    supports_system_prompt: bool
    supports_conversation_history: bool
    multimodal: Optional[bool] = False
    last_error: Optional[str] = None

class GenerationRequest(BaseModel):
    model_name: str
    prompt: str
    system_prompt: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    max_tokens: int = 500
    temperature: float = 0.7
    
class GenerationResponse(BaseModel):
    response: str
    model_name: str
    provider: str
    generation_time_ms: int
    input_tokens: int
    output_tokens: int
    total_tokens: int

class ComparisonRequest(BaseModel):
    models: List[str]
    prompt: str
    system_prompt: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    max_tokens: int = 500
    temperature: float = 0.7

class ComparisonResponse(BaseModel):
    prompt: str
    results: Dict[str, Any]
    comparison_id: Optional[str] = None

class ModelComparisonCreate(BaseModel):
    prompt_text: str
    phi2_response_id: str
    external_response_id: str
    winner: str  # 'phi2', 'external', 'tie'
    human_preference: Optional[str] = None

@router.get("/models", response_model=List[ExternalModelInfo])
async def get_external_models(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of available external models"""
    try:
        model_info = external_model_manager.get_all_model_info()
        
        external_models = []
        for model_name, info in model_info.items():
            external_models.append(ExternalModelInfo(
                model_name=model_name,
                provider=info["provider"],
                is_available=info["is_available"],
                is_configured=info["is_configured"],
                context_length=info["context_length"],
                supports_system_prompt=info["supports_system_prompt"],
                supports_conversation_history=info["supports_conversation_history"],
                multimodal=info.get("multimodal", False),
                last_error=info.get("last_error")
            ))
        
        return external_models
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get external models: {str(e)}"
        )

@router.post("/generate", response_model=GenerationResponse)
async def generate_external_response(
    request: GenerationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Generate response from external model"""
    try:
        # Sanitize input
        prompt = sanitize_input(request.prompt)
        system_prompt = sanitize_input(request.system_prompt) if request.system_prompt else None
        
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty"
            )
        
        # Check if model is available
        if not external_model_manager.is_model_available(request.model_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_name} is not available"
            )
        
        # Generate response
        response_data = await external_model_manager.generate_response(
            model_name=request.model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=request.conversation_history,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return GenerationResponse(
            response=response_data["response"],
            model_name=response_data["model_name"],
            provider=response_data["provider"],
            generation_time_ms=response_data["generation_time_ms"],
            input_tokens=response_data["input_tokens"],
            output_tokens=response_data["output_tokens"],
            total_tokens=response_data["total_tokens"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )

@router.post("/compare", response_model=ComparisonResponse)
async def compare_models(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Compare responses from multiple models"""
    try:
        # Sanitize input
        prompt = sanitize_input(request.prompt)
        system_prompt = sanitize_input(request.system_prompt) if request.system_prompt else None
        
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty"
            )
        
        # Check if models are available
        available_models = external_model_manager.get_available_models()
        invalid_models = [m for m in request.models if m not in available_models]
        
        if invalid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Models not available: {invalid_models}"
            )
        
        # Generate responses from all models
        results = await external_model_manager.compare_models(
            model_names=request.models,
            prompt=prompt,
            system_prompt=system_prompt,
            conversation_history=request.conversation_history,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Add Phi-2 response for comparison
        try:
            from ..core.llm_manager import phi2_manager
            phi2_response = await phi2_manager.generate_response_async(
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_history=request.conversation_history,
                generation_params={
                    "max_new_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )
            
            results["phi-2"] = {
                "response": phi2_response["response"],
                "model_name": "phi-2",
                "provider": "local",
                "generation_time_ms": phi2_response["generation_time_ms"],
                "input_tokens": phi2_response["input_tokens"],
                "output_tokens": phi2_response["output_tokens"],
                "total_tokens": phi2_response["total_tokens"]
            }
        except Exception as e:
            results["phi-2"] = {"error": f"Phi-2 generation failed: {str(e)}"}
        
        return ComparisonResponse(
            prompt=prompt,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )

@router.post("/comparison/submit")
async def submit_comparison_result(
    comparison_data: ModelComparisonCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit comparison result between models"""
    try:
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
        
        return {
            "success": True,
            "comparison_id": str(comparison.id),
            "message": "Comparison result submitted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit comparison: {str(e)}"
        )

@router.get("/comparisons", response_model=List[Dict[str, Any]])
async def get_model_comparisons(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """Get model comparison history"""
    try:
        comparisons = db.query(ModelComparison).order_by(
            ModelComparison.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        comparison_data = []
        for comparison in comparisons:
            # Get response details
            phi2_response = db.query(ModelResponse).filter(
                ModelResponse.id == comparison.phi2_response_id
            ).first()
            
            external_response = db.query(ModelResponse).filter(
                ModelResponse.id == comparison.external_response_id
            ).first()
            
            comparison_data.append({
                "id": str(comparison.id),
                "prompt_text": comparison.prompt_text,
                "winner": comparison.winner,
                "human_preference": comparison.human_preference,
                "created_at": comparison.created_at.isoformat(),
                "phi2_response": {
                    "id": str(phi2_response.id),
                    "response_text": phi2_response.response_text[:200] + "..." if len(phi2_response.response_text) > 200 else phi2_response.response_text,
                    "generation_time_ms": phi2_response.generation_time_ms
                } if phi2_response else None,
                "external_response": {
                    "id": str(external_response.id),
                    "model_name": external_response.model_name,
                    "response_text": external_response.response_text[:200] + "..." if len(external_response.response_text) > 200 else external_response.response_text,
                    "generation_time_ms": external_response.generation_time_ms
                } if external_response else None
            })
        
        return comparison_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get comparisons: {str(e)}"
        )

@router.get("/stats")
async def get_external_model_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get statistics about external model usage"""
    try:
        # Get comparison statistics
        total_comparisons = db.query(ModelComparison).count()
        
        # Win rate by model
        phi2_wins = db.query(ModelComparison).filter(
            ModelComparison.winner == "phi2"
        ).count()
        
        external_wins = db.query(ModelComparison).filter(
            ModelComparison.winner == "external"
        ).count()
        
        ties = db.query(ModelComparison).filter(
            ModelComparison.winner == "tie"
        ).count()
        
        # External model usage
        external_responses = db.query(ModelResponse).filter(
            ModelResponse.model_name != "phi-2"
        ).all()
        
        model_usage = {}
        for response in external_responses:
            model_name = response.model_name
            if model_name not in model_usage:
                model_usage[model_name] = {
                    "count": 0,
                    "avg_generation_time": 0,
                    "total_tokens": 0
                }
            
            model_usage[model_name]["count"] += 1
            model_usage[model_name]["avg_generation_time"] += response.generation_time_ms
            model_usage[model_name]["total_tokens"] += response.token_count
        
        # Calculate averages
        for model_name, stats in model_usage.items():
            if stats["count"] > 0:
                stats["avg_generation_time"] = stats["avg_generation_time"] / stats["count"]
        
        return {
            "total_comparisons": total_comparisons,
            "phi2_win_rate": phi2_wins / total_comparisons if total_comparisons > 0 else 0,
            "external_win_rate": external_wins / total_comparisons if total_comparisons > 0 else 0,
            "tie_rate": ties / total_comparisons if total_comparisons > 0 else 0,
            "model_usage": model_usage,
            "available_models": external_model_manager.get_available_models(),
            "configured_models": len(external_model_manager.models)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )

@router.post("/refresh")
async def refresh_external_models(
    current_user: User = Depends(get_current_active_user)
):
    """Refresh external model connections"""
    try:
        external_model_manager.refresh_models()
        
        available_models = external_model_manager.get_available_models()
        
        return {
            "success": True,
            "message": "External models refreshed successfully",
            "available_models": available_models,
            "total_available": len(available_models)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh models: {str(e)}"
        )