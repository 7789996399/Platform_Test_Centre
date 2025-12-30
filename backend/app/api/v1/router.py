"""
TRUST Platform - API Router Aggregator
"""
from fastapi import APIRouter
from .scribe import router as scribe_router
from .cerner import router as cerner_router
from .ml import router as ml_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(scribe_router)
api_router.include_router(cerner_router)
api_router.include_router(ml_router)