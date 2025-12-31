"""
TRUST Platform Audit Service
============================
Logging all AI document reviews to Azure PostgreSQL.
PHI is automatically de-identified before storage using Azure Health Data AI Services.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import Optional, Dict, Any
from ..models.database import AuditLog
from .deid_client import deid_client
import logging

logger = logging.getLogger(__name__)


async def redact_phi_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact PHI from analysis results before storing in audit log.
    Protects patient privacy while maintaining audit trail integrity.
    """
    redacted = results.copy()
    
    # Fields that may contain PHI
    phi_fields = [
        "note_text", "transcript", "source_transcript", 
        "clinical_text", "raw_text", "patient_name"
    ]
    
    for field in phi_fields:
        if field in redacted and redacted[field]:
            try:
                redacted[field] = await deid_client.redact_for_audit(str(redacted[field]))
            except Exception as e:
                logger.warning(f"Failed to redact {field}: {e}")
                redacted[field] = "[REDACTION_FAILED]"
    
    # Redact PHI in nested claim objects
    if "claims" in redacted and isinstance(redacted["claims"], list):
        for claim in redacted["claims"]:
            if isinstance(claim, dict) and "text" in claim:
                try:
                    claim["text"] = await deid_client.redact_for_audit(str(claim["text"]))
                except Exception as e:
                    logger.warning(f"Failed to redact claim: {e}")
                    claim["text"] = "[REDACTED]"
    
    # Mark as de-identified
    redacted["_phi_redacted"] = True
    redacted["_redaction_timestamp"] = datetime.utcnow().isoformat()
    
    return redacted


async def log_analysis(
    db: AsyncSession,
    note_id: str,
    patient_id: str,
    results: Dict[str, Any],
    physician_id: Optional[str] = None,
    physician_action: Optional[str] = None,
    skip_deidentification: bool = False,
) -> AuditLog:
    """
    Log an AI document analysis to the audit trail.
    PHI is automatically de-identified before storage.
    """
    
    # De-identify PHI before storing
    if not skip_deidentification:
        safe_results = await redact_phi_from_results(results)
    else:
        safe_results = results
    
    # Hash the patient_id for privacy (keep original for lookup)
    # In production, consider storing only hashed IDs
    
    audit = AuditLog(
        note_id=note_id,
        patient_id=patient_id,  # Consider hashing in production
        encounter_type=safe_results.get("encounter_type"),
        confidence_score=safe_results.get("confidence_score"),
        semantic_entropy=safe_results.get("semantic_entropy"),
        review_level=safe_results.get("review_recommendation"),
        claims_total=safe_results.get("claims_total", 0),
        claims_verified=safe_results.get("verified", 0),
        claims_unverified=safe_results.get("unverified", 0),
        claims_contradicted=safe_results.get("contradicted", 0),
        hallucinations_detected=safe_results.get("hallucinations", 0),
        ehr_verified=safe_results.get("ehr_verified", 0),
        ehr_not_found=safe_results.get("ehr_not_found", 0),
        ehr_contradicted=safe_results.get("ehr_contradicted", 0),
        physician_id=physician_id,
        physician_action=physician_action,
        analysis_details=safe_results,
    )
    
    db.add(audit)
    await db.commit()
    await db.refresh(audit)
    
    logger.info(f"Audit log created: {audit.id} (PHI redacted: {not skip_deidentification})")
    
    return audit


async def get_audit_logs(
    db: AsyncSession,
    patient_id: Optional[str] = None,
    limit: int = 100,
) -> list:
    """Retrieve audit logs, optionally filtered by patient."""
    
    query = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)
    
    if patient_id:
        query = query.where(AuditLog.patient_id == patient_id)
    
    result = await db.execute(query)
    return result.scalars().all()
