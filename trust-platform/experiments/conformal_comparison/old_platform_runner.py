#!/usr/bin/env python3
"""
OLD Platform Runner
===================

Wraps the OLD EHR-First validation pipeline (Semantic Entropy only, no conformal).

Based on patterns from: backend/app/services/trust_scribe_ehr_first_validation.py

This runner:
1. Checks claim against EHR data
2. Computes Semantic Entropy from multiple responses
3. Assigns risk level based on EHR status + SE
4. Returns raw measurements (no interpretation)
"""

import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from enum import Enum

# Import test case structure
try:
    from experiments.conformal_comparison.test_cases import TestCase, ClaimType
except ImportError:
    from test_cases import TestCase, ClaimType


# =============================================================================
# ENUMS (from OLD platform)
# =============================================================================

class EHRVerificationStatus(str, Enum):
    """Result of EHR verification check."""
    VERIFIED = "verified"
    CONTRADICTION = "contradiction"
    NOT_FOUND = "not_found"
    NOT_CHECKABLE = "not_checkable"


class SemanticEntropyLevel(str, Enum):
    """Interpretation of SE score."""
    LOW = "low"       # SE < 0.3
    MEDIUM = "medium" # SE 0.3-0.6
    HIGH = "high"     # SE > 0.6


class RiskLevel(str, Enum):
    """Final risk assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# RESULT DATA CLASS
# =============================================================================

@dataclass
class OldPlatformResult:
    """Result from OLD platform processing."""
    case_id: str
    ehr_status: str
    semantic_entropy: float
    se_level: str
    risk_level: str
    requires_review: bool
    computation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# SEMANTIC ENTROPY CALCULATION
# =============================================================================

def compute_semantic_entropy(responses: List[str]) -> float:
    """
    Compute semantic entropy from multiple LLM responses.

    This is a simplified version of the actual SE calculation.
    In production, responses are clustered by semantic meaning,
    then entropy is computed over cluster distribution.

    For this experiment, we use a proxy based on response variation.
    """
    if not responses:
        return 0.0

    # Extract "semantic categories" from responses
    # Simple heuristic: categorize by key words
    categories = []
    for resp in responses:
        resp_lower = resp.lower()
        if "verified" in resp_lower or "confirmed" in resp_lower or "correct" in resp_lower:
            categories.append("verified")
        elif "contradiction" in resp_lower or "incorrect" in resp_lower or "error" in resp_lower:
            categories.append("contradiction")
        elif "uncertain" in resp_lower or "cannot" in resp_lower or "review" in resp_lower:
            categories.append("uncertain")
        else:
            categories.append("other")

    # Count category frequencies
    counts = {}
    for cat in categories:
        counts[cat] = counts.get(cat, 0) + 1

    # Compute entropy: -sum(p * log(p))
    n = len(categories)
    if n == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log(p + 1e-10)

    # Normalize to roughly 0-1 range (log(4) ≈ 1.39 for 4 categories)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    normalized_entropy = entropy / max(max_entropy, 1.0)

    return round(normalized_entropy, 4)


def classify_se_level(se: float) -> SemanticEntropyLevel:
    """Classify SE score into level."""
    if se < 0.3:
        return SemanticEntropyLevel.LOW
    elif se < 0.6:
        return SemanticEntropyLevel.MEDIUM
    else:
        return SemanticEntropyLevel.HIGH


# =============================================================================
# EHR VERIFICATION
# =============================================================================

def verify_against_ehr(case: TestCase) -> EHRVerificationStatus:
    """
    Verify claim against EHR data.

    In production, this calls FHIR APIs. Here we use the test case's
    pre-computed expected status based on whether the claim matches EHR.
    """
    if not case.ehr_data_available:
        return EHRVerificationStatus.NOT_FOUND

    # Use the expected status from test case generation
    # (which was computed based on whether claim matches EHR)
    if case.expected_ehr_status == "verified":
        return EHRVerificationStatus.VERIFIED
    elif case.expected_ehr_status == "contradiction":
        return EHRVerificationStatus.CONTRADICTION
    elif case.expected_ehr_status == "not_found":
        return EHRVerificationStatus.NOT_FOUND
    else:
        return EHRVerificationStatus.NOT_CHECKABLE


# =============================================================================
# RISK ASSESSMENT
# =============================================================================

def compute_risk_level(
    ehr_status: EHRVerificationStatus,
    se_level: SemanticEntropyLevel,
    claim_type: ClaimType
) -> RiskLevel:
    """
    Compute risk level based on EHR status and SE.

    OLD Platform Risk Matrix:

    | EHR Status    | SE Level | Risk Level |
    |---------------|----------|------------|
    | verified      | any      | LOW        |
    | contradiction | LOW      | CRITICAL   | ← Confident hallucinator!
    | contradiction | MEDIUM   | HIGH       |
    | contradiction | HIGH     | MEDIUM     | ← Uncertain, less dangerous
    | not_found     | LOW      | MEDIUM     | ← New info, confident
    | not_found     | MEDIUM   | MEDIUM     |
    | not_found     | HIGH     | HIGH       | ← New info, uncertain
    """
    if ehr_status == EHRVerificationStatus.VERIFIED:
        return RiskLevel.LOW

    if ehr_status == EHRVerificationStatus.CONTRADICTION:
        # Contradiction + Low SE = Confident hallucinator (most dangerous)
        if se_level == SemanticEntropyLevel.LOW:
            return RiskLevel.CRITICAL
        elif se_level == SemanticEntropyLevel.MEDIUM:
            return RiskLevel.HIGH
        else:
            return RiskLevel.MEDIUM

    if ehr_status == EHRVerificationStatus.NOT_FOUND:
        # Not in EHR - could be new info or missed data
        if se_level == SemanticEntropyLevel.HIGH:
            return RiskLevel.HIGH
        else:
            return RiskLevel.MEDIUM

    # NOT_CHECKABLE
    if se_level == SemanticEntropyLevel.HIGH:
        return RiskLevel.HIGH
    elif se_level == SemanticEntropyLevel.MEDIUM:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def requires_physician_review(risk_level: RiskLevel) -> bool:
    """Determine if physician review is needed."""
    return risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


# =============================================================================
# MAIN RUNNER
# =============================================================================

class OldPlatformRunner:
    """
    Runner for OLD platform (EHR-First with SE only).

    Implements the pipeline:
    1. Verify claim against EHR
    2. Compute SE from multiple responses
    3. Assign risk based on EHR status + SE
    """

    def __init__(self):
        self.name = "OLD_PLATFORM"
        self.version = "1.0"

    def process_case(self, case: TestCase) -> OldPlatformResult:
        """
        Process a single test case through OLD platform.

        Returns raw measurements only.
        """
        start_time = time.perf_counter()

        # Step 1: Verify against EHR
        ehr_status = verify_against_ehr(case)

        # Step 2: Compute semantic entropy
        # In OLD platform, SE is computed for ALL claims (expensive)
        # But in EHR-First optimization, only for contradictions
        # For this experiment, we compute for all to have consistent measurements
        se = compute_semantic_entropy(case.multiple_responses)
        se_level = classify_se_level(se)

        # Step 3: Compute risk level
        risk_level = compute_risk_level(ehr_status, se_level, case.claim_type)

        # Step 4: Determine if review needed
        review_needed = requires_physician_review(risk_level)

        end_time = time.perf_counter()
        computation_ms = (end_time - start_time) * 1000

        return OldPlatformResult(
            case_id=case.case_id,
            ehr_status=ehr_status.value,
            semantic_entropy=se,
            se_level=se_level.value,
            risk_level=risk_level.value,
            requires_review=review_needed,
            computation_time_ms=round(computation_ms, 3),
        )

    def process_batch(self, cases: List[TestCase]) -> List[OldPlatformResult]:
        """Process multiple cases."""
        return [self.process_case(case) for case in cases]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from test_cases import generate_test_cases

    print("OLD Platform Runner Test")
    print("=" * 50)

    # Generate a few test cases
    cases = generate_test_cases(n_cases=10, seed=42)

    runner = OldPlatformRunner()

    print(f"\nProcessing {len(cases)} cases...\n")

    for case in cases[:5]:
        result = runner.process_case(case)

        print(f"{result.case_id}:")
        print(f"  Claim: {case.claim_text[:50]}...")
        print(f"  Ground truth: {'correct' if case.ground_truth_correct else 'incorrect'}")
        print(f"  EHR status: {result.ehr_status}")
        print(f"  SE: {result.semantic_entropy:.4f} ({result.se_level})")
        print(f"  Risk: {result.risk_level}")
        print(f"  Review needed: {result.requires_review}")
        print(f"  Time: {result.computation_time_ms:.3f} ms")
        print()
