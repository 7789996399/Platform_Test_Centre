"""
TRUST Platform - Scribe API Models
===================================
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


# =============================================================
# ENUMS
# =============================================================

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ReviewTier(str, Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    CONTRADICTED = "contradicted"
    PARTIAL = "partial"


# =============================================================
# REQUEST MODELS
# =============================================================

class MedicationInput(BaseModel):
    """Single medication entry."""
    name: str
    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None


class ScribeNoteInput(BaseModel):
    """AI scribe note submitted for analysis."""
    note_id: str
    patient_id: str
    patient_name: Optional[str] = None
    encounter_type: Optional[str] = None
    encounter_date: Optional[str] = None
    sections: Dict[str, str | List[MedicationInput]]
    source_transcript: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "note_id": "note-001",
                "patient_id": "patient-12345",
                "patient_name": "Scott Jackson",
                "encounter_type": "Pre-operative Assessment",
                "sections": {
                    "chief_complaint": "67-year-old male for pre-op assessment",
                    "medications": [
                        {"name": "Metoprolol", "dose": "50mg", "frequency": "BID"}
                    ],
                    "allergies": "NKDA"
                },
                "source_transcript": "Good morning Mr. Jackson..."
            }
        }


# =============================================================
# RESPONSE MODELS
# =============================================================

class ClaimResponse(BaseModel):
    """Single claim in analysis response."""
    claim_id: str
    text: str
    claim_type: str
    risk_level: str
    section: str


class VerificationResponse(BaseModel):
    """Verification result for a claim."""
    status: VerificationStatus
    transcript_match: Optional[str] = None
    match_score: float
    explanation: str


class EntropyResponse(BaseModel):
    """Semantic entropy result."""
    entropy: float
    n_clusters: int
    risk_level: RiskLevel


class UncertaintyResponse(BaseModel):
    """Uncertainty quantification result."""
    confidence: float
    consistency: float
    calibrated_confidence: float
    review_tier: ReviewTier
    flags: List[str]


class ClaimAnalysisResponse(BaseModel):
    """Complete analysis of a single claim."""
    rank: int
    claim: ClaimResponse
    verification: VerificationResponse
    entropy: Optional[EntropyResponse] = None
    uncertainty: UncertaintyResponse
    priority_score: float


class ReviewBurdenResponse(BaseModel):
    """Review burden statistics."""
    total_claims: int
    brief_review: int
    standard_review: int
    detailed_review: int
    estimated_review_seconds: int
    time_saved_percent: float


class AnalysisSummaryResponse(BaseModel):
    """Summary of note analysis."""
    total_claims: int
    verified: int
    needs_review: int
    contradictions: int
    compute_saved_percent: float
    high_priority_count: int


class NoteAnalysisResponse(BaseModel):
    """Complete analysis response."""
    note_id: str
    patient_id: str
    analyzed_at: datetime
    overall_risk: RiskLevel
    summary: AnalysisSummaryResponse
    review_burden: ReviewBurdenResponse
    review_queue: List[ClaimAnalysisResponse]


class AuditLogResponse(BaseModel):
    """Audit log entry."""
    timestamp: datetime
    note_id: str
    patient_id: str
    trust_version: str
    overall_risk: RiskLevel
    compliance_frameworks: List[str]


# =============================================================
# HEALTH CHECK
# =============================================================

class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    timestamp: datetime
