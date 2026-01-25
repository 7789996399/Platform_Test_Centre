"""
TRUST Clinical Risk Adapter

Domain-specific adapter for clinical risk prediction AI governance including:
- Sepsis risk prediction (qSOFA, NEWS)
- Readmission risk
- Deterioration early warning
- Input validation for vital signs and labs
- Mock EHR integration for testing
"""

from adapters.clinical_risk.clinical_risk_adapter import (
    ClinicalRiskAdapter,
    RiskScoreType,
    VitalSign,
    LabValue,
    VITAL_SIGN_RANGES,
    LAB_VALUE_RANGES,
    RISK_ACTION_THRESHOLDS,
)

__all__ = [
    'ClinicalRiskAdapter',
    'RiskScoreType',
    'VitalSign',
    'LabValue',
    'VITAL_SIGN_RANGES',
    'LAB_VALUE_RANGES',
    'RISK_ACTION_THRESHOLDS',
]
