"""
TRUST Platform - Diagnostic Governance Module
==============================================

Governance pipeline for diagnostic AI systems (Radiology AI, Pathology AI).

Key Characteristics of Diagnostic AI:
    - Outputs are classifications or segmentations (not free-form text)
    - Model confidence != clinical reliability
    - Prior imaging/context is critical for interpretation
    - Attention maps can reveal model failures

The Conformal-First Flow:
    1. Run conformal prediction to get prediction SET (not single class)
    2. Apply domain-specific context (prior imaging, clinical history)
    3. Validate attention maps against expected patterns
    4. Output calibrated prediction with coverage guarantee

Why Conformal Prediction for Diagnostics?
    A radiology AI saying "95% confident pneumonia" is NOT the same as
    "95% of the time when I say pneumonia, I'm right." Conformal prediction
    provides the latter - actual coverage guarantees:

    "The true finding will be in this prediction set with >= 90% probability"

    This transforms uncertain point predictions into actionable intervals.

Example Use Cases:
    - Chest X-ray interpretation
    - CT scan lesion detection
    - Mammography screening
    - Pathology slide analysis
    - ECG interpretation
"""

from diagnostic_governance.pipeline import (
    DiagnosticGovernancePipeline,
    DiagnosticGovernanceResult,
    DiagnosticPrediction,
)

from diagnostic_governance.attention_validator import (
    AttentionValidator,
    AttentionValidationResult,
    AttentionRegion,
)

from diagnostic_governance.prior_context import (
    PriorContextIntegrator,
    PriorStudy,
    ContextualFinding,
)

__all__ = [
    # Pipeline
    'DiagnosticGovernancePipeline',
    'DiagnosticGovernanceResult',
    'DiagnosticPrediction',
    # Attention Validation
    'AttentionValidator',
    'AttentionValidationResult',
    'AttentionRegion',
    # Prior Context
    'PriorContextIntegrator',
    'PriorStudy',
    'ContextualFinding',
]
