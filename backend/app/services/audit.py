"""
TRUST Platform Audit Service
============================
Logging all AI document reviews to Azure PostgreSQL.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import Optional, Dict, Any

from ..models.database import AuditLog


async def log_analysis(
    db: AsyncSession,
    note_id: str,
    patient_id: str,
    results: Dict[str, Any],
    physician_id: Optional[str] = None,
    physician_action: Optional[str] = None,
) -> AuditLog:
    """Log an AI document analysis to the audit trail."""
    
    audit = AuditLog(
        note_id=note_id,
        patient_id=patient_id,
        encounter_type=results.get("encounter_type"),
        confidence_score=results.get("confidence_score"),
        semantic_entropy=results.get("semantic_entropy"),
        review_level=results.get("review_recommendation"),
        claims_total=results.get("claims_total", 0),
        claims_verified=results.get("verified", 0),
        claims_unverified=results.get("unverified", 0),
        claims_contradicted=results.get("contradicted", 0),
        hallucinations_detected=results.get("hallucinations", 0),
        ehr_verified=results.get("ehr_verified", 0),
        ehr_not_found=results.get("ehr_not_found", 0),
        ehr_contradicted=results.get("ehr_contradicted", 0),
        physician_id=physician_id,
        physician_action=physician_action,
        analysis_details=results,
    )
    
    db.add(audit)
    await db.commit()
    await db.refresh(audit)
    
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
