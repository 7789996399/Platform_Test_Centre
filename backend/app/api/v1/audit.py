"""
TRUST Platform - Audit Logging API
==================================
Endpoints for logging and retrieving user activity.
"""
from fastapi import APIRouter, Request, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
from ...db import get_db
from ...models.database import UserActivityLog

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditLogCreate(BaseModel):
    """Schema for creating audit log entries."""
    user_id: str
    user_email: str
    user_name: Optional[str] = None
    action: str  # LOGIN, LOGOUT, VIEW_DASHBOARD, APPROVE_DOC, REJECT_DOC, etc.
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[dict] = None


class AuditLogResponse(BaseModel):
    """Schema for audit log response."""
    id: int
    timestamp: datetime
    user_id: str
    user_email: str
    user_name: Optional[str]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    ip_address: Optional[str]
    details: Optional[dict]

    class Config:
        from_attributes = True


@router.post("/log", response_model=dict)
async def create_audit_log(
    log_entry: AuditLogCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Create a new audit log entry."""
    
    # Get IP address from request
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", None)
    
    # Create log entry
    db_log = UserActivityLog(
        user_id=log_entry.user_id,
        user_email=log_entry.user_email,
        user_name=log_entry.user_name,
        action=log_entry.action,
        resource_type=log_entry.resource_type,
        resource_id=log_entry.resource_id,
        ip_address=ip_address,
        user_agent=user_agent,
        details=log_entry.details
    )
    
    db.add(db_log)
    await db.commit()
    
    return {"status": "logged", "action": log_entry.action}


@router.get("/logs", response_model=list[AuditLogResponse])
async def get_audit_logs(
    db: AsyncSession = Depends(get_db),
    user_email: Optional[str] = Query(None, description="Filter by user email"),
    action: Optional[str] = Query(None, description="Filter by action type"),
    days: int = Query(7, description="Number of days to look back"),
    limit: int = Query(100, description="Max records to return")
):
    """Retrieve audit logs with optional filtering."""
    
    # Build query
    query = select(UserActivityLog)
    
    # Apply filters
    if user_email:
        query = query.where(UserActivityLog.user_email == user_email)
    if action:
        query = query.where(UserActivityLog.action == action)
    
    # Time filter
    cutoff = datetime.utcnow() - timedelta(days=days)
    query = query.where(UserActivityLog.timestamp >= cutoff)
    
    # Order and limit
    query = query.order_by(desc(UserActivityLog.timestamp)).limit(limit)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return logs


@router.get("/summary", response_model=dict)
async def get_audit_summary(
    db: AsyncSession = Depends(get_db),
    days: int = Query(7, description="Number of days to summarize")
):
    """Get summary statistics of audit logs."""
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    # Get all logs in period
    query = select(UserActivityLog).where(UserActivityLog.timestamp >= cutoff)
    result = await db.execute(query)
    logs = result.scalars().all()
    
    # Calculate summary
    action_counts = {}
    unique_users = set()
    
    for log in logs:
        action_counts[log.action] = action_counts.get(log.action, 0) + 1
        if log.user_email:
            unique_users.add(log.user_email)
    
    return {
        "period_days": days,
        "total_events": len(logs),
        "unique_users": len(unique_users),
        "actions": action_counts
    }
