"""
TRUST Platform - FastAPI Backend
=================================
Auditing AI. Protecting Patients. Empowering Physicians.
Main entry point for the API server.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from .config import settings
from .api.v1.router import api_router

# =============================================================
# APP INITIALIZATION
# =============================================================
app = FastAPI(
    title=settings.project_name,
    description="""
## Healthcare AI Governance Platform
**Transparent, Responsible, Unbiased, Safe, Traceable**

TRUST provides independent oversight of AI-generated clinical documentation.

### Key Features:
- Hallucination Detection (Semantic Entropy)
- Uncertainty Quantification
- Source Verification
- EHR Verification (Cerner FHIR)
- Audit Logging
    """,
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# =============================================================
# MIDDLEWARE
# =============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# ROUTES
# =============================================================
app.include_router(api_router)

@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint."""
    return {
        "name": settings.project_name,
        "version": settings.version,
        "environment": settings.environment,
        "status": "healthy",
        "docs": "/docs",
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.version,
        "environment": settings.environment,
    }
