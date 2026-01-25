"""
TRUST Platform - Predictive Governance Module
=============================================

Governance pipeline for predictive AI systems (Risk Scores, Early Warning).

Key Characteristics of Predictive AI:
    - Outputs are numerical scores or risk levels
    - Point predictions hide uncertainty
    - Garbage in = garbage out (input quality critical)
    - Predictions must be calibrated to actual outcomes

The Input-Validation-First Flow:
    1. Validate inputs BEFORE expensive model inference
    2. Reject or flag invalid/implausible inputs
    3. Run conformal prediction for calibrated intervals
    4. Track outcomes for continuous calibration

Why Input Validation First?
    Predictive models are only as good as their inputs. A sepsis prediction
    model given implausible vitals (BP = 0/0) will produce garbage. By
    validating inputs first, we:
    - Save compute on invalid inputs
    - Catch data quality issues early
    - Prevent nonsensical predictions from reaching clinicians

Example Use Cases:
    - Sepsis risk scores (qSOFA, NEWS)
    - Readmission risk prediction
    - Deterioration early warning (MEWS)
    - Mortality prediction (APACHE, SAPS)
    - Length of stay estimation
"""

from predictive_governance.pipeline import (
    PredictiveGovernancePipeline,
    PredictiveGovernanceResult,
    CalibratedPrediction,
)

from predictive_governance.input_validator import (
    InputValidator,
    InputValidationResult,
    ValidationRule,
)

from predictive_governance.outcome_tracker import (
    OutcomeTracker,
    TrackedPrediction,
    CalibrationStats,
)

__all__ = [
    # Pipeline
    'PredictiveGovernancePipeline',
    'PredictiveGovernanceResult',
    'CalibratedPrediction',
    # Input Validation
    'InputValidator',
    'InputValidationResult',
    'ValidationRule',
    # Outcome Tracking
    'OutcomeTracker',
    'TrackedPrediction',
    'CalibrationStats',
]
