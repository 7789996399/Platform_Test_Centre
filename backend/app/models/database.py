"""
TRUST Platform Database Models
==============================
SQLAlchemy models for audit logging and governance tracking.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class AuditLog(Base):
    """Audit trail for all AI document reviews."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Document info
    note_id = Column(String(100), index=True)
    patient_id = Column(String(100), index=True)
    encounter_type = Column(String(50))
    
    # Analysis results
    confidence_score = Column(Float)
    semantic_entropy = Column(Float)
    review_level = Column(String(20))  # brief, standard, detailed
    
    # Verification counts
    claims_total = Column(Integer)
    claims_verified = Column(Integer)
    claims_unverified = Column(Integer)
    claims_contradicted = Column(Integer)
    hallucinations_detected = Column(Integer)
    
    # EHR verification
    ehr_verified = Column(Integer)
    ehr_not_found = Column(Integer)
    ehr_contradicted = Column(Integer)
    
    # Physician action
    physician_id = Column(String(100))
    physician_action = Column(String(20))  # approved, modified, rejected
    physician_notes = Column(Text)
    
    # Full results (JSON)
    analysis_details = Column(JSON)


class AIModel(Base):
    """Registry of AI models under governance."""
    __tablename__ = "ai_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True)
    model_type = Column(String(50))  # predictive, generative, radiology
    vendor = Column(String(100))
    version = Column(String(50))
    status = Column(String(20), default="active")  # active, review, suspended
    
    # Performance metrics
    accuracy = Column(Float)
    last_evaluated = Column(DateTime(timezone=True))
    
    # Compliance scores (JSON: {framework: score})
    compliance_scores = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ComplianceCheck(Base):
    """Compliance evaluation records."""
    __tablename__ = "compliance_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    model_id = Column(Integer, index=True)
    framework = Column(String(50))  # FDA_GMLP, EU_AI_ACT, etc.
    score = Column(Float)
    passed = Column(Boolean)
    findings = Column(JSON)

class MLAnalysisLog(Base):
    """Audit log for ML service analyses."""
    __tablename__ = "ml_analysis_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Analysis type
    analysis_type = Column(String(50))  # semantic_entropy, uncertainty, hallucination, embeddings
    
    # Input
    input_text = Column(Text)
    input_params = Column(JSON)
    
    # Results
    result_score = Column(Float)  # Primary score (entropy, uncertainty, etc.)
    confidence = Column(Float)
    review_level = Column(String(20))  # BRIEF, STANDARD, DETAILED
    
    # Full response
    full_response = Column(JSON)
    
    # Performance
    processing_time_ms = Column(Float)
    ml_service_version = Column(String(20))
    
    # Context (optional)
    user_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    note_id = Column(String(100), nullable=True)   
