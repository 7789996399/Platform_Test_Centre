"""
TRUST Platform - Generative Governance Module
==============================================

Governance pipeline for generative AI systems (AI Scribes, Chatbots, Summarizers).

Key Characteristics of Generative AI:
    - Free-form text output (not constrained to predefined classes)
    - Can fabricate plausible-sounding but incorrect information
    - Confidence is orthogonal to correctness (confident hallucinator problem)
    - Outputs must be verified against authoritative sources

The EHR-First Flow:
    1. Extract verifiable claims from AI output
    2. Verify claims against EHR/authoritative sources FIRST (fast, cheap)
    3. Only run semantic entropy on UNVERIFIED claims (expensive)
    4. Flag "confident hallucinators" (low entropy + contradicts source)

This flow achieves ~80% compute reduction vs always running semantic entropy,
while catching the most dangerous error pattern: confident but wrong.

Example Use Cases:
    - AI Medical Scribes (documenting patient encounters)
    - Clinical Decision Support chatbots
    - Medical report summarization
    - Patient communication assistants
"""

from generative_governance.pipeline import (
    GenerativeGovernancePipeline,
    GenerativeGovernanceResult,
    ClaimVerificationSummary,
)

from generative_governance.claim_extractor import (
    GenerativeClaimExtractor,
    ExtractedClaim,
    ExtractionConfig,
)

__all__ = [
    # Pipeline
    'GenerativeGovernancePipeline',
    'GenerativeGovernanceResult',
    'ClaimVerificationSummary',
    # Claim Extraction
    'GenerativeClaimExtractor',
    'ExtractedClaim',
    'ExtractionConfig',
]
