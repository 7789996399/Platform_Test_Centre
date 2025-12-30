"""
ML Service Client
Connects API Backend to ML Microservice
"""
import httpx
import os
from typing import Dict, List, Optional

# ML Service URL - Azure deployed service
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "https://trust-ml-service.azurewebsites.net")

# Timeout settings (ML can take time to load models)
TIMEOUT = httpx.Timeout(60.0, connect=10.0)


async def get_ml_health() -> Dict:
    """Check ML service health"""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{ML_SERVICE_URL}/health")
        response.raise_for_status()
        return response.json()


async def calculate_semantic_entropy(
    text: str,
    num_samples: int = 5,
    model: str = "local"
) -> Dict:
    """
    Call ML service to calculate semantic entropy.
    
    Returns:
        {
            "entropy": float (0-1, higher = more uncertain),
            "confidence": float (0-1, higher = more confident),
            "samples": list of generated samples,
            "processing_time_ms": float
        }
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{ML_SERVICE_URL}/analyze/semantic-entropy",
            json={
                "text": text,
                "num_samples": num_samples,
                "model": model
            }
        )
        response.raise_for_status()
        return response.json()


async def generate_embeddings(texts: List[str]) -> Dict:
    """
    Call ML service to generate text embeddings.
    
    Returns:
        {
            "embeddings": list of embedding vectors,
            "model": model name used,
            "dimensions": embedding dimensions
        }
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{ML_SERVICE_URL}/analyze/embeddings",
            json={"texts": texts}
        )
        response.raise_for_status()
        return response.json()


async def detect_hallucination(claim: str, context: str) -> Dict:
    """
    Call ML service to detect if a claim is a hallucination.
    
    Returns:
        {
            "is_hallucination": bool,
            "confidence": float,
            "reasoning": str
        }
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{ML_SERVICE_URL}/analyze/hallucination",
            json={
                "claim": claim,
                "context": context
            }
        )
        response.raise_for_status()
        return response.json()


async def calculate_uncertainty(
    text: str,
    method: str = "ensemble"
) -> Dict:
    """
    Call ML service to calculate uncertainty quantification.
    (Will be added to ML service next)
    
    Returns:
        {
            "uncertainty": float,
            "calibrated_confidence": float,
            "method": str
        }
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{ML_SERVICE_URL}/analyze/uncertainty",
            json={
                "text": text,
                "method": method
            }
        )
        response.raise_for_status()
        return response.json()
    