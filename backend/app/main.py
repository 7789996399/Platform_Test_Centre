"""
TRUST Platform - FastAPI Backend
================================
Auditing AI. Protecting Patients. Empowering Physicians.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TRUST Platform API",
    description="Healthcare AI Governance - Transparent, Responsible, Unbiased, Safe, Traceable",
    version="0.1.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "TRUST Platform API",
        "tagline": "Auditing AI. Protecting Patients. Empowering Physicians.",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}
