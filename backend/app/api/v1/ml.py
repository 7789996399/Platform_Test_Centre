"""
ML Analysis API Endpoints
Exposes ML service capabilities to frontend with audit logging
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from ...services.ml_client import (
    get_ml_health,
    calculate_semantic_entropy,
    generate_embeddings,
    detect_hallucination,
    calculate_uncertainty
)
from ...db import get_db
from ...models.database import MLAnalysisLog

router = APIRouter(prefix="/ml", tags=["ML Analysis"])


# ============== Request/Response Models ==============

class SemanticEntropyRequest(BaseModel):
    text: str
    num_samples: int = 5
    model: str = "local"
    note_id: Optional[str] = None

class HallucinationRequest(BaseModel):
    claim: str
    context: str
    note_id: Optional[str] = None

class EmbeddingsRequest(BaseModel):
    texts: List[str]

class UncertaintyRequest(BaseModel):
    text: str
    method: str = "calibrated"
    note_id: Optional[str] = None


# ============== Logging Helper ==============

async def log_ml_analysis(
    db: AsyncSession,
    analysis_type: str,
    input_text: str,
    input_params: dict,
    result: dict,
    note_id: Optional[str] = None
):
    """Log ML analysis to database for audit trail."""
    try:
        log_entry = MLAnalysisLog(
            analysis_type=analysis_type,
            input_text=input_text[:1000],
            input_params=input_params,
            result_score=result.get("entropy") or result.get("uncertainty") or result.get("confidence"),
            confidence=result.get("confidence") or result.get("calibrated_confidence"),
            review_level=result.get("review_level"),
            full_response=result,
            processing_time_ms=result.get("processing_time_ms"),
            ml_service_version="0.1.0",
            note_id=note_id
        )
        db.add(log_entry)
        await db.commit()
    except Exception as e:
        print(f"Warning: Failed to log ML analysis: {e}")


# ============== Endpoints ==============

@router.get("/health")
async def ml_health():
    """Check ML service health"""
    try:
        result = await get_ml_health()
        return {"status": "connected", "ml_service": result}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ML service unavailable: {str(e)}")


@router.post("/semantic-entropy")
async def semantic_entropy(request: SemanticEntropyRequest, db: AsyncSession = Depends(get_db)):
    """Calculate semantic entropy for hallucination detection."""
    try:
        result = await calculate_semantic_entropy(
            text=request.text,
            num_samples=request.num_samples,
            model=request.model
        )
        
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
        
        await log_ml_analysis(db, "semantic_entropy", request.text,
            {"num_samples": request.num_samples, "model": request.model}, result, request.note_id)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hallucination")
async def hallucination_detection(request: HallucinationRequest, db: AsyncSession = Depends(get_db)):
    """Detect if a claim is supported by the given context."""
    try:
        result = await detect_hallucination(claim=request.claim, context=request.context)
        result["review_level"] = "DETAILED" if result.get("is_hallucination") else "BRIEF"
        
        await log_ml_analysis(db, "hallucination", 
            f"CLAIM: {request.claim}\nCONTEXT: {request.context}", {}, result, request.note_id)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings")
async def generate_embeddings_endpoint(request: EmbeddingsRequest):
    """Generate text embeddings for similarity comparison"""
    try:
        return await generate_embeddings(request.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/uncertainty")
async def uncertainty_quantification(request: UncertaintyRequest, db: AsyncSession = Depends(get_db)):
    """Calculate calibrated uncertainty for AI outputs."""
    try:
        result = await calculate_uncertainty(text=request.text, method=request.method)
        
        await log_ml_analysis(db, "uncertainty", request.text,
            {"method": request.method}, result, request.note_id)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Audit Endpoints ==============

@router.get("/logs")
async def get_ml_logs(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Get recent ML analysis logs."""
    try:
        from sqlalchemy import select
        query = select(MLAnalysisLog).order_by(MLAnalysisLog.timestamp.desc()).limit(limit)
        result = await db.execute(query)
        logs = result.scalars().all()
        
        return {
            "count": len(logs),
            "logs": [
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "analysis_type": log.analysis_type,
                    "result_score": log.result_score,
                    "review_level": log.review_level,
                    "processing_time_ms": log.processing_time_ms,
                    "note_id": log.note_id
                }
                for log in logs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/summary")
async def get_ml_logs_summary(db: AsyncSession = Depends(get_db)):
    """Get summary statistics for ML analyses."""
    try:
        from sqlalchemy import select, func
        
        total_result = await db.execute(select(func.count(MLAnalysisLog.id)))
        total_count = total_result.scalar()
        
        type_result = await db.execute(
            select(MLAnalysisLog.analysis_type, func.count(MLAnalysisLog.id))
            .group_by(MLAnalysisLog.analysis_type))
        by_type = {row[0]: row[1] for row in type_result}
        
        level_result = await db.execute(
            select(MLAnalysisLog.review_level, func.count(MLAnalysisLog.id))
            .group_by(MLAnalysisLog.review_level))
        by_level = {row[0]: row[1] for row in level_result}
        
        avg_result = await db.execute(select(func.avg(MLAnalysisLog.processing_time_ms)))
        avg_time = avg_result.scalar()
        
        return {
            "total_analyses": total_count,
            "by_type": by_type,
            "by_review_level": by_level,
            "avg_processing_time_ms": float(avg_time) if avg_time else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))