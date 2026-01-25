"""
TRUST Healthcare Adapter

Domain-specific adapter for healthcare AI governance including:
- Clinical terminology validation
- Drug interaction checking
- HIPAA compliance verification
- Medical claim source attribution
"""

from .healthcare_adapter import (
    HealthcareAdapter,
    HealthcareClaimType,
    HealthcareVerificationResult,
    MedicationComponents,
    ReviewLevel,
    AUTHORITATIVE_DRUG_SOURCES,
    AUTHORITATIVE_CLINICAL_SOURCES,
    COMMON_ALLERGENS,
    MEDICATION_PATTERNS,
    VITAL_SIGN_PATTERNS,
    ALLERGY_PATTERNS,
    EXAM_FINDING_PATTERNS,
)

__all__ = [
    'HealthcareAdapter',
    'HealthcareClaimType',
    'HealthcareVerificationResult',
    'MedicationComponents',
    'ReviewLevel',
    'AUTHORITATIVE_DRUG_SOURCES',
    'AUTHORITATIVE_CLINICAL_SOURCES',
    'COMMON_ALLERGENS',
    'MEDICATION_PATTERNS',
    'VITAL_SIGN_PATTERNS',
    'ALLERGY_PATTERNS',
    'EXAM_FINDING_PATTERNS',
]
