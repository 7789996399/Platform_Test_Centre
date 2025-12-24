"""
TRUST Platform - API Router Aggregator
=======================================
Combines all v1 API routes.
"""

from fastapi import APIRouter
from .scribe import router as scribe_router

# Main v1 router
api_router = APIRouter(prefix="/api/v1")

# Include all route modules
api_router.include_router(scribe_router)

# Future routes will be added here:
# api_router.include_router(predictive_router)
# api_router.include_router(radiology_router)
