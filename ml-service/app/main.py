"""
TRUST ML Service
Handles heavy AI workloads: semantic entropy, embeddings, hallucination detection

Model Loading Strategy:
- Models are pre-loaded at startup to avoid cold-start latency
- Currently using DeBERTa-v3-small for P1v2 App Service (140MB, fits in 3.5GB RAM)
- TODO: Upgrade to Azure ML Endpoint with DeBERTa-v3-large for pilot
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# STARTUP: Pre-load models to avoid cold-start latency
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ML models at startup"""
    logger.info("🚀 Starting TRUST ML Service...")
    
    # Pre-load entailment model (DeBERTa-v3-small)
    try:
        from .semantic_entropy import EntailmentClassifier
        logger.info("📦 Pre-loading entailment model...")
        EntailmentClassifier.preload_model()
        logger.info("✅ Entailment model loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-load entailment model: {e}")
    
    # Pre-load sentence transformer for embeddings
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("📦 Pre-loading sentence transformer...")
        global sentence_model
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Sentence transformer loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-load sentence transformer: {e}")
    
    logger.info("🟢 TRUST ML Service ready!")
    
    yield  # App runs here
    
    logger.info("🔴 Shutting down TRUST ML Service...")


app = FastAPI(
    title="TRUST ML Service",
    description="AI/ML processing for TRUST Platform - Semantic Entropy & Hallucination Detection",
    version="0.2.0",
    lifespan=lifespan
)

# Global model reference (set during startup)
sentence_model = None

# CORS - allow API backend to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trust-api-phc.azurewebsites.net",
        "https://api.trustplatform.ca",
        "https://www.trustplatform.ca",
        "https://trustplatform.ca",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Request/Response Models ==============

class SemanticEntropyRequest(BaseModel):
    text: str
    context: str = ""
    num_samples: int = 5
    model: str = "openai"  # "openai", "anthropic", "mock"

class SemanticEntropyResponse(BaseModel):
    entropy: float
    normalized_entropy: float
    confidence: float
    num_clusters: int
    num_samples: int
    cluster_sizes: List[int]
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
    from .semantic_entropy import EntailmentClassifier
    
    models_loaded = []
    if sentence_model is not None:
        models_loaded.append("sentence-transformers")
    if EntailmentClassifier.is_loaded():
        models_loaded.append("deberta-v3-small-mnli")
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.2.0",
        "models_loaded": models_loaded
    }

# ============== Semantic Entropy ==============

@app.post("/analyze/semantic-entropy", response_model=SemanticEntropyResponse)
async def calculate_semantic_entropy_endpoint(request: SemanticEntropyRequest):
    """
    Calculate semantic entropy for hallucination detection.
    
    Based on Farquhar et al. methodology:
    1. Generate N responses to the same prompt
    2. Cluster by bidirectional entailment
    3. Calculate entropy across clusters
    
    Lower entropy = more consistent = likely accurate
    Higher entropy = inconsistent = possible hallucination
    
    NOTE: Only called for claims that FAILED EHR verification (EHR-First approach)
    """
    import time
    start = time.time()
    
    try:
        from .semantic_entropy import calculate_semantic_entropy
        
        result = await calculate_semantic_entropy(
            prompt=request.text,
            context=request.context,
            num_samples=request.num_samples,
            model=request.model
        )
        
        processing_time = (time.time() - start) * 1000
        
        return {
            "entropy": result.entropy,
            "normalized_entropy": result.normalized_entropy,
            "confidence": result.confidence,
            "num_clusters": result.num_clusters,
            "num_samples": result.num_samples,
            "cluster_sizes": result.cluster_sizes,
            "samples": result.samples,
            "processing_time_ms": processing_time
        }
    except Exception as e:
        logger.error(f"Semantic entropy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Embeddings ==============

@app.post("/analyze/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate text embeddings for similarity comparison"""
    try:
        global sentence_model
        if sentence_model is None:
            from sentence_transformers import SentenceTransformer
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = sentence_model.encode(request.texts).tolist()
        
        return {
            "embeddings": embeddings,
            "model": "all-MiniLM-L6-v2",
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Hallucination Detection ==============

class HallucinationRequest(BaseModel):
    claim: str
    context: str
    ehr_verification: Optional[dict] = None
    
class HallucinationResponse(BaseModel):
    is_hallucination: bool
    risk_level: str
    confidence: float
    semantic_entropy: float
    ehr_status: str
    reasoning: str
    review_level: str

@app.post("/analyze/hallucination", response_model=HallucinationResponse)
async def detect_hallucination_endpoint(request: HallucinationRequest):
    """
    Detect if a claim is a hallucination using multi-layer approach:
    1. EHR Verification (should be done before calling this endpoint)
    2. Semantic Entropy
    3. Confident Hallucinator Detection
    
    NOTE: This endpoint should only be called for claims that FAILED EHR verification
    """
    try:
        from .hallucination import detect_hallucination
        
        result = await detect_hallucination(
            claim=request.claim,
            context=request.context,
            ehr_verification=request.ehr_verification,
            calculate_entropy=True
        )
        
        return {
            "is_hallucination": result.is_hallucination,
            "risk_level": result.risk_level.value,
            "confidence": result.confidence,
            "semantic_entropy": result.semantic_entropy,
            "ehr_status": result.ehr_status.value,
            "reasoning": result.reasoning,
            "review_level": result.review_level
        }
    except Exception as e:
        logger.error(f"Hallucination detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Batch Analysis ==============

class BatchAnalysisRequest(BaseModel):
    claims: List[dict]  # [{"text": "...", "claim_type": "medication"}, ...]
    context: str
    ehr_data: Optional[dict] = None

@app.post("/analyze/batch")
async def analyze_claims_batch(request: BatchAnalysisRequest):
    """
    Analyze multiple claims from an AI-generated note.
    Uses EHR-First approach: only runs SE on unverified claims.
    """
    try:
        from .hallucination import analyze_claims
        
        result = await analyze_claims(
            claims=request.claims,
            context=request.context,
            ehr_data=request.ehr_data
        )
        
        return result
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
