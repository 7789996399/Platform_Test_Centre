"""
TRUST Platform - Cerner API Routes
===================================
Endpoints for Cerner FHIR data access.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from ...services.cerner import (
    get_patient_context,
    search_patients,
    PatientContext
)


router = APIRouter(prefix="/cerner", tags=["Cerner FHIR"])


class PatientResponse(BaseModel):
    patient_id: str
    name: str
    birth_date: str
    gender: str
    conditions: List[str]
    medications: List[str]
    allergies: List[str]


class PatientListItem(BaseModel):
    id: str
    name: str
    birthDate: Optional[str]
    gender: Optional[str]


@router.get("/patients", response_model=List[PatientListItem])
async def list_patients(count: int = 10):
    """List available test patients from Cerner sandbox."""
    patients = search_patients(count)
    return patients


@router.get("/patient/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str):
    """
    Get complete patient context for AI verification.
    
    Returns demographics, conditions, medications, and allergies.
    """
    ctx = get_patient_context(patient_id)
    
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    return PatientResponse(
        patient_id=ctx.patient_id,
        name=ctx.name,
        birth_date=ctx.birth_date,
        gender=ctx.gender,
        conditions=ctx.conditions,
        medications=ctx.medications,
        allergies=ctx.allergies
    )


@router.get("/health")
async def cerner_health():
    """Check Cerner connection status."""
    try:
        patients = search_patients(1)
        return {
            "status": "connected",
            "sandbox": "fhir-open.cerner.com",
            "test_patients_available": len(patients) > 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
