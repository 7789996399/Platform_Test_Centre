"""
TRUST Platform - Scribe API Routes
===================================
Endpoints for AI scribe note governance.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from datetime import datetime

from ...models.scribe import (
    ScribeNoteInput,
    NoteAnalysisResponse,
    ClaimAnalysisResponse,
    ClaimResponse,
    VerificationResponse,
    EntropyResponse,
    UncertaintyResponse,
    AnalysisSummaryResponse,
    ReviewBurdenResponse,
    AuditLogResponse,
    RiskLevel,
    ReviewTier,
    VerificationStatus,
    HealthResponse
)
from ...core.routing import analyze_note, format_review_queue_for_display, generate_audit_log
from ...core.claim_extraction import extract_claims_from_note
from ...core.ehr_verification import verify_note_against_ehr, EHRVerificationStatus

router = APIRouter(prefix="/scribe", tags=["AI Scribe Governance"])


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def convert_note_input_to_dict(note_input: ScribeNoteInput) -> dict:
    """Convert Pydantic model to dict format expected by core modules."""
    
    # Convert medication list if present
    sections = {}
    for key, value in note_input.sections.items():
        if key == "medications" and isinstance(value, list):
            sections[key] = [
                {
                    "name": med.name,
                    "dose": med.dose or "",
                    "route": med.route or "",
                    "frequency": med.frequency or ""
                }
                for med in value
            ]
        else:
            sections[key] = value
    
    return {
        "note_id": note_input.note_id,
        "patient": {
            "id": note_input.patient_id,
            "name": note_input.patient_name or "Unknown"
        },
        "encounter": {
            "type": note_input.encounter_type or "Unknown",
            "date": note_input.encounter_date or datetime.now().isoformat()
        },
        "ai_scribe_output": {
            "sections": sections
        },
        "source_transcript": note_input.source_transcript
    }


def convert_analysis_to_response(analysis, note_input: ScribeNoteInput) -> NoteAnalysisResponse:
    """Convert internal analysis object to API response."""
    
    review_queue = []
    for i, ca in enumerate(analysis.review_queue):
        review_queue.append(ClaimAnalysisResponse(
            rank=i + 1,
            claim=ClaimResponse(
                claim_id=ca.claim.id,
                text=ca.claim.text,
                claim_type=ca.claim.claim_type.value,
                risk_level=ca.claim.risk_level,
                section=ca.claim.section
            ),
            verification=VerificationResponse(
                status=VerificationStatus(ca.verification.status.value),
                transcript_match=ca.verification.transcript_match,
                match_score=ca.verification.match_score,
                explanation=ca.verification.explanation
            ),
            entropy=EntropyResponse(
                entropy=ca.entropy.entropy,
                n_clusters=ca.entropy.n_clusters,
                risk_level=RiskLevel(ca.entropy.risk_level)
            ) if ca.entropy else None,
            uncertainty=UncertaintyResponse(
                confidence=ca.uncertainty.confidence,
                consistency=ca.uncertainty.consistency,
                calibrated_confidence=ca.uncertainty.calibrated_confidence,
                review_tier=ReviewTier(ca.uncertainty.review_tier.value),
                flags=ca.uncertainty.flags
            ),
            priority_score=ca.priority_score
        ))
    
    return NoteAnalysisResponse(
        note_id=analysis.note_id,
        patient_id=analysis.patient_id,
        analyzed_at=analysis.analyzed_at,
        overall_risk=RiskLevel(analysis.overall_risk),
        summary=AnalysisSummaryResponse(
            total_claims=analysis.summary["total_claims"],
            verified=analysis.summary["verified"],
            needs_review=analysis.summary["needs_review"],
            contradictions=analysis.summary["contradictions"],
            compute_saved_percent=analysis.summary["compute_saved_percent"],
            high_priority_count=analysis.summary["high_priority_count"]
        ),
        review_burden=ReviewBurdenResponse(
            total_claims=analysis.review_burden["total_claims"],
            brief_review=analysis.review_burden["brief_review"],
            standard_review=analysis.review_burden["standard_review"],
            detailed_review=analysis.review_burden["detailed_review"],
            estimated_review_seconds=analysis.review_burden["estimated_review_seconds"],
            time_saved_percent=analysis.review_burden["time_saved_percent"]
        ),
        review_queue=review_queue
    )


# =============================================================
# ENDPOINTS
# =============================================================

@router.post("/analyze", response_model=NoteAnalysisResponse)
async def analyze_scribe_note(note_input: ScribeNoteInput):
    """
    Analyze an AI-generated scribe note for hallucinations and uncertainty.
    
    This is the main endpoint that:
    1. Extracts claims from the note
    2. Verifies claims against source transcript
    3. Calculates semantic entropy for unverified claims
    4. Assigns review tiers based on uncertainty
    5. Returns prioritized review queue
    
    Returns:
        Complete analysis with review queue sorted by priority
    """
    try:
        # Convert input to internal format
        note_dict = convert_note_input_to_dict(note_input)
        transcript = note_input.source_transcript
        
        # Run analysis pipeline
        analysis = analyze_note(note_dict, transcript, run_entropy=True)
        
        # Convert to response format
        response = convert_analysis_to_response(analysis, note_input)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/quick", response_model=NoteAnalysisResponse)
async def analyze_scribe_note_quick(note_input: ScribeNoteInput):
    """
    Quick analysis without semantic entropy (faster, less accurate).
    
    Use for real-time feedback during note generation.
    Full analysis should be run before physician sign-off.
    """
    try:
        note_dict = convert_note_input_to_dict(note_input)
        transcript = note_input.source_transcript
        
        # Run without entropy (much faster)
        analysis = analyze_note(note_dict, transcript, run_entropy=False)
        
        response = convert_analysis_to_response(analysis, note_input)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")


@router.post("/audit-log", response_model=AuditLogResponse)
async def create_audit_log(note_input: ScribeNoteInput):
    """
    Generate audit log entry for a note analysis.
    
    Required for regulatory compliance:
    - FDA GMLP
    - EU AI Act
    - Health Canada SaMD
    - WHO AI Ethics
    """
    try:
        note_dict = convert_note_input_to_dict(note_input)
        transcript = note_input.source_transcript
        
        analysis = analyze_note(note_dict, transcript, run_entropy=True)
        audit = generate_audit_log(analysis)
        
        return AuditLogResponse(
            timestamp=datetime.fromisoformat(audit["timestamp"]),
            note_id=audit["note_id"],
            patient_id=audit["patient_id"],
            trust_version=audit["trust_version"],
            overall_risk=RiskLevel(analysis.overall_risk),
            compliance_frameworks=audit["compliance_frameworks"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit log generation failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the scribe analysis service is healthy."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now()
    )
@router.post("/verify-ehr/{patient_id}")
async def verify_against_ehr(patient_id: str, note_input: ScribeNoteInput):
    """
    Verify AI scribe claims against real Cerner EHR data.
    
    This is the TRUST advantage - comparing AI output against
    the actual medical record, not just the transcript!
    """
    try:
        # Convert input and extract claims
        note_dict = convert_note_input_to_dict(note_input)
        claims = extract_claims_from_note(note_dict)
        
        # Verify against EHR
        ehr_result = verify_note_against_ehr(claims, patient_id)
        
        if ehr_result["status"] == "error":
            raise HTTPException(status_code=404, detail=ehr_result["error"])
        
        # Format results
        formatted_results = []
        for r in ehr_result["results"]:
            formatted_results.append({
                "claim_text": r.claim.text,
                "claim_type": r.claim.claim_type.value,
                "status": r.status.value,
                "ehr_match": r.ehr_match,
                "confidence": r.confidence,
                "explanation": r.explanation
            })
        
        return {
            "patient_id": ehr_result["patient_id"],
            "patient_name": ehr_result["patient_name"],
            "total_claims": ehr_result["total_claims"],
            "verified": ehr_result["verified"],
            "contradicted": ehr_result["contradicted"],
            "not_in_ehr": ehr_result["not_in_ehr"],
            "ehr_medications_count": ehr_result["ehr_medications_count"],
            "ehr_allergies_count": ehr_result["ehr_allergies_count"],
            "results": formatted_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EHR verification failed: {str(e)}")
