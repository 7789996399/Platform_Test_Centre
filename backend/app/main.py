"""
TRUST Platform - FastAPI Backend
=================================
Auditing AI. Protecting Patients. Empowering Physicians.

Main entry point for the API server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from .api.v1.router import api_router

# =============================================================
# APP INITIALIZATION
# =============================================================

app = FastAPI(
    title="TRUST Platform API",
    description="""
## Healthcare AI Governance Platform

**Transparent, Responsible, Unbiased, Safe, Traceable**

TRUST provides independent oversight of AI-generated clinical documentation,
implementing semantic entropy for hallucination detection and uncertainty
quantification for calibrated confidence scoring.

### Key Features:
- 🔍 **Hallucination Detection** - Semantic entropy analysis (Paper 1)
- 📊 **Uncertainty Quantification** - Calibrated confidence (Paper 2)
- ✅ **Source Verification** - Fast transcript matching
- 📋 **Review Prioritization** - Tiered physician oversight
- 📝 **Audit Logging** - Regulatory compliance

### Compliance Frameworks:
- FDA Good Machine Learning Practice (GMLP)
- EU AI Act
- Health Canada SaMD Guidance
- WHO AI Ethics Guidelines
    """,
    version="0.1.0",
    contact={
        "name": "TRUST Platform",
        "email": "trust@example.com",
    },
    license_info={
        "name": "Proprietary",
    },
)

# =============================================================
# MIDDLEWARE
# =============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React dev server
        "http://localhost:5173",      # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================
# ROUTES
# =============================================================

# Include API routes
app.include_router(api_router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint with API information."""
    return {
        "name": "TRUST Platform API",
        "tagline": "Auditing AI. Protecting Patients. Empowering Physicians.",
        "version": "0.1.0",
        "status": "healthy",
        "docs": "/docs",
        "endpoints": {
            "analyze": "/api/v1/scribe/analyze",
            "quick_analyze": "/api/v1/scribe/analyze/quick",
            "audit_log": "/api/v1/scribe/audit-log",
            "health": "/api/v1/scribe/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Global health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "services": {
            "api": "healthy",
            "scribe_analysis": "healthy"
        }
    }
