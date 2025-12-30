"""
ML Analysis API Endpoints
Exposes ML service capabilities to frontend
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ...services.ml_client import (
    get_ml_health,
    calculate_semantic_entropy,
    generate_embeddings,
    detect_hallucination,
    calculate_uncertainty
)

router = APIRouter(prefix="/ml", tags=["ML Analysis"])


# ============== Request/Response Models ==============

class SemanticEntropyRequest(BaseModel):
    text: str
    num_samples: int = 5
    model: str = "local"

class HallucinationRequest(BaseModel):
    claim: str
    context: str

class EmbeddingsRequest(BaseModel):
    texts: List[str]

class UncertaintyRequest(BaseModel):
    text: str
    method: str = "ensemble"


# ============== Endpoints ==============

@router.get("/health")
async def ml_health():
    """Check ML service health"""
    try:
        result = await get_ml_health()
        return {"status": "connected", "ml_service": result}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"ML service unavailable: {str(e)}"
        )


@router.post("/semantic-entropy")
async def semantic_entropy(request: SemanticEntropyRequest):
    """
    Calculate semantic entropy for hallucination detection.
    
    - Low entropy (< 0.3): High confidence, likely accurate
    - Medium entropy (0.3-0.7): Moderate confidence, review recommended
    - High entropy (> 0.7): Low confidence, likely hallucination
    """
    try:
        result = await calculate_semantic_entropy(
            text=request.text,
            num_samples=request.num_samples,
            model=request.model
        )
        
        # Add interpretation
        entropy = result["entropy"]
        if entropy < 0.3:
            result["interpretation"] = "HIGH_CONFIDENCE"
            result["review_level"] = "BRIEF"
        elif entropy < 0.7:
            result["interpretation"] = "MODERATE_CONFIDENCE"
            result["review_level"] = "STANDARD"
        else:
            result["interpretation"] = "LOW_CONFIDENCE"
            result["review_level"] = "DETAILED"
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hallucination")
async def hallucination_detection(request: HallucinationRequest):
    """
    Detect if a claim is supported by the given context.
    Uses Natural Language Inference (NLI).
    """
    try:
        return await detect_hallucination(
            claim=request.claim,
            context=request.context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings")
async def generate_embeddings(request: EmbeddingsRequest):
    """Generate text embeddings for similarity comparison"""
    try:
        return await generate_embeddings(request.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/uncertainty")
async def uncertainty_quantification(request: UncertaintyRequest):
    """
    Calculate calibrated uncertainty for AI outputs.
    Based on Paper 2 methodology.
    """
    try:
        return await calculate_uncertainty(
            text=request.text,
            method=request.method
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))