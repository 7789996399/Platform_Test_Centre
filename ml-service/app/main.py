"""
TRUST ML Service
Handles heavy AI workloads: semantic entropy, embeddings, hallucination detection
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from .uncertainty import calculate_uncertainty, medical_adjusted_uncertainty
import os

app = FastAPI(
    title="TRUST ML Service",
    description="AI/ML processing for TRUST Platform",
    version="0.1.0"
)

# CORS - allow API backend to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://api.trustplatform.ca",
        "https://www.trustplatform.ca",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Request/Response Models ==============

class SemanticEntropyRequest(BaseModel):
    text: str
    num_samples: int = 5
    model: str = "local"  # "local", "openai", "anthropic"

class SemanticEntropyResponse(BaseModel):
    entropy: float
    confidence: float
    samples: List[str]
    processing_time_ms: float

class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    models_loaded: List[str]

# ============== Health Check ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "models_loaded": ["sentence-transformers"]  # Will update when models load
    }

# ============== Semantic Entropy ==============

@app.post("/analyze/semantic-entropy", response_model=SemanticEntropyResponse)
async def calculate_semantic_entropy(request: SemanticEntropyRequest):
    """
    Calculate semantic entropy for hallucination detection.
    Lower entropy = more consistent = likely accurate
    Higher entropy = inconsistent = possible hallucination
    """
    import time
    start = time.time()
    
    try:
        from .semantic_entropy import calculate_entropy
        
        result = await calculate_entropy(
            text=request.text,
            num_samples=request.num_samples,
            model=request.model
        )
        
        processing_time = (time.time() - start) * 1000
        
        return {
            "entropy": result["entropy"],
            "confidence": result["confidence"],
            "samples": result["samples"],
            "processing_time_ms": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============== Embeddings ==============

@app.post("/analyze/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate text embeddings for similarity comparison"""
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(request.texts).tolist()
        
        return {
            "embeddings": embeddings,
            "model": "all-MiniLM-L6-v2",
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============== Hallucination Detection ==============

class HallucinationRequest(BaseModel):
    claim: str
    context: str
    
class HallucinationResponse(BaseModel):
    is_hallucination: bool
    confidence: float
    reasoning: str

@app.post("/analyze/hallucination", response_model=HallucinationResponse)
async def detect_hallucination(request: HallucinationRequest):
    """
    Detect if a claim is a hallucination given the context.
    Uses NLI (Natural Language Inference) approach.
    """
    try:
        from .hallucination import detect
        
        result = await detect(
            claim=request.claim,
            context=request.context
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ============== Uncertainty Quantification ==============

class UncertaintyRequest(BaseModel):
    text: str
    method: str = "calibrated"  # "ensemble", "token_probability", "calibrated"
    num_samples: int = 5
    medical_context: bool = True

class UncertaintyResponse(BaseModel):
    uncertainty: float
    calibrated_confidence: float
    method: str
    review_level: str
    details: dict

@app.post("/analyze/uncertainty", response_model=UncertaintyResponse)
async def analyze_uncertainty(request: UncertaintyRequest):
    """
    Calculate calibrated uncertainty for AI-generated text.
    Based on Paper 2 methodology.
    
    Methods:
    - ensemble: Multiple sample variance (most robust)
    - token_probability: Linguistic hedging analysis
    - calibrated: Combined approach with calibration curve
    """
    import time
    start = time.time()
    
    try:
        result = await calculate_uncertainty(
            text=request.text,
            method=request.method,
            num_samples=request.num_samples
        )
        
        # Apply medical adjustment if requested
        if request.medical_context:
            result["uncertainty"] = await medical_adjusted_uncertainty(
                request.text, 
                result["uncertainty"]
            )
            result["calibrated_confidence"] = 1.0 - result["uncertainty"]
            result["details"]["medical_adjusted"] = True
        
        # Determine review level based on uncertainty
        uncertainty = result["uncertainty"]
        if uncertainty < 0.2:
            review_level = "BRIEF"
        elif uncertainty < 0.4:
            review_level = "BRIEF"
        elif uncertainty < 0.6:
            review_level = "STANDARD"
        else:
            review_level = "DETAILED"
        
        result["review_level"] = review_level
        result["details"]["processing_time_ms"] = (time.time() - start) * 1000
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)