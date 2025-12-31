"""
TRUST Platform - Cerner API Routes
===================================
Endpoints for Cerner FHIR data access and AI document detection.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from ...services.cerner import (
    get_patient_context,
    search_patients,
    get_document_references,
    get_document_by_id,
    PatientContext
)
from ...services.cerner_mapper import CernerDocumentMapper, TRUSTDocument

router = APIRouter(prefix="/cerner", tags=["Cerner FHIR"])


# =============================================================================
# RESPONSE MODELS
# =============================================================================

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


class DocumentResponse(BaseModel):
    """Response model for mapped clinical documents."""
    document_id: str
    patient_id: str
    encounter_id: Optional[str]
    note_type: str
    author: str
    author_type: str  # ai_scribe, physician, nurse, unknown
    is_ai_generated: bool
    review_level: str  # brief, standard, detailed
    created_at: str
    note_preview: str  # First 200 chars
    cerner_resource_id: str


class DocumentDetailResponse(DocumentResponse):
    """Full document with complete note text."""
    note_text: str


class DocumentListResponse(BaseModel):
    """Response for document list endpoint."""
    documents: List[DocumentResponse]
    total_count: int
    ai_generated_count: int
    human_authored_count: int


# =============================================================================
# PATIENT ENDPOINTS (existing)
# =============================================================================

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


# =============================================================================
# DOCUMENT ENDPOINTS (new!)
# =============================================================================

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    count: int = Query(10, ge=1, le=50, description="Number of documents to fetch")
):
    """
    Fetch and analyze clinical documents from Cerner.
    
    This endpoint:
    1. Fetches DocumentReference resources from Cerner FHIR
    2. Maps them to TRUST format
    3. DETECTS AI-GENERATED CONTENT automatically
    4. Returns documents with author classification
    
    AI-generated documents (author_type = "ai_scribe") require governance review.
    """
    # Fetch raw documents from Cerner
    raw_docs = get_document_references(patient_id=patient_id, count=count)
    
    # Map each document
    mapper = CernerDocumentMapper()
    documents = []
    ai_count = 0
    human_count = 0
    
    for raw_doc in raw_docs:
        result = mapper.map(raw_doc)
        
        if result.success:
            doc = result.document
            
            # Count by author type
            if doc.is_ai_generated():
                ai_count += 1
            else:
                human_count += 1
            
            # Create response
            documents.append(DocumentResponse(
                document_id=doc.document_id,
                patient_id=doc.patient_id,
                encounter_id=doc.encounter_id,
                note_type=doc.note_type,
                author=doc.author,
                author_type=doc.author_type,
                is_ai_generated=doc.is_ai_generated(),
                review_level=doc.review_level,
                created_at=doc.created_at.isoformat(),
                note_preview=doc.note_text[:200] + "..." if len(doc.note_text) > 200 else doc.note_text,
                cerner_resource_id=doc.cerner_resource_id,
            ))
    
    return DocumentListResponse(
        documents=documents,
        total_count=len(documents),
        ai_generated_count=ai_count,
        human_authored_count=human_count,
    )


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: str):
    """
    Get a specific document with full note text.
    
    Use this to view the complete clinical note for review.
    """
    # Remove our prefix if present
    cerner_id = document_id.replace("cerner_", "")
    
    raw_doc = get_document_by_id(cerner_id)
    
    if not raw_doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    mapper = CernerDocumentMapper()
    result = mapper.map(raw_doc)
    
    if not result.success:
        raise HTTPException(status_code=500, detail=f"Failed to map document: {result.errors}")
    
    doc = result.document
    
    return DocumentDetailResponse(
        document_id=doc.document_id,
        patient_id=doc.patient_id,
        encounter_id=doc.encounter_id,
        note_type=doc.note_type,
        author=doc.author,
        author_type=doc.author_type,
        is_ai_generated=doc.is_ai_generated(),
        review_level=doc.review_level,
        created_at=doc.created_at.isoformat(),
        note_preview=doc.note_text[:200] + "..." if len(doc.note_text) > 200 else doc.note_text,
        cerner_resource_id=doc.cerner_resource_id,
        note_text=doc.note_text,
    )


@router.get("/documents/ai-generated", response_model=List[DocumentResponse])
async def list_ai_documents(
    patient_id: Optional[str] = Query(None, description="Filter by patient ID"),
    count: int = Query(20, ge=1, le=50)
):
    """
    List ONLY AI-generated documents requiring governance review.
    
    This is a convenience endpoint for the physician review dashboard.
    Returns only documents where author_type = "ai_scribe".
    """
    # Fetch and map
    raw_docs = get_document_references(patient_id=patient_id, count=count)
    mapper = CernerDocumentMapper()
    
    ai_documents = []
    
    for raw_doc in raw_docs:
        result = mapper.map(raw_doc)
        
        if result.success and result.document.is_ai_generated():
            doc = result.document
            ai_documents.append(DocumentResponse(
                document_id=doc.document_id,
                patient_id=doc.patient_id,
                encounter_id=doc.encounter_id,
                note_type=doc.note_type,
                author=doc.author,
                author_type=doc.author_type,
                is_ai_generated=True,
                review_level=doc.review_level,
                created_at=doc.created_at.isoformat(),
                note_preview=doc.note_text[:200] + "..." if len(doc.note_text) > 200 else doc.note_text,
                cerner_resource_id=doc.cerner_resource_id,
            ))
    
    return ai_documents


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/health")
async def cerner_health():
    """Check Cerner connection and document access."""
    try:
        # Test patient access
        patients = search_patients(1)
        patient_ok = len(patients) > 0
        
        # Test document access
        docs = get_document_references(count=1)
        docs_ok = True  # Even empty is OK
        
        return {
            "status": "connected",
            "sandbox": "fhir-open.cerner.com",
            "patient_access": patient_ok,
            "document_access": docs_ok,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }