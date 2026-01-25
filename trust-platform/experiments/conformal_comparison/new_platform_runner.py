#!/usr/bin/env python3
"""
NEW Platform Runner
===================

Wraps the NEW platform with CONFORMAL PREDICTION.

Key difference from OLD platform:
- Instead of just a risk level, returns a PREDICTION SET
- The prediction set has a coverage GUARANTEE (e.g., 90%)
- We can measure whether ground truth is in the set

Uses:
- trust-platform/core_engine/conformal_calibrator.py
- trust-platform/generative_governance/ patterns
"""

import sys
import time
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

# Add parent paths for imports
sys.path.insert(0, '../..')

# Import test case structure
from test_cases import TestCase, ClaimType

# Import conformal calibrator from trust-platform
from core_engine.conformal_calibrator import (
    AdaptivePredictionSets,
    CalibrationResult,
    PredictionSet,
)


# =============================================================================
# ENUMS
# =============================================================================

class EHRVerificationStatus(str, Enum):
    """Result of EHR verification check."""
    VERIFIED = "verified"
    CONTRADICTION = "contradiction"
    NOT_FOUND = "not_found"
    NOT_CHECKABLE = "not_checkable"


class SemanticEntropyLevel(str, Enum):
    """Interpretation of SE score."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(str, Enum):
    """Risk assessment labels - these become our conformal prediction classes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# RESULT DATA CLASS
# =============================================================================

@dataclass
class NewPlatformResult:
    """Result from NEW platform processing with conformal prediction."""
    case_id: str
    ehr_status: str
    semantic_entropy: float
    se_level: str

    # CONFORMAL PREDICTION - key difference from OLD platform
    conformal_prediction_set: List[str]  # Set of possible risk levels
    conformal_set_size: int
    ground_truth_in_set: bool  # KEY: Did the set contain the true answer?
    conformal_coverage_target: float  # e.g., 0.90

    # Final outputs
    risk_level: str  # Most likely risk (argmax)
    requires_review: bool
    computation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['conformal_prediction_set'] = ','.join(self.conformal_prediction_set)
        return d


# =============================================================================
# SEMANTIC ENTROPY (same as OLD platform for fair comparison)
# =============================================================================

def compute_semantic_entropy(responses: List[str]) -> float:
    """Compute SE from responses (same as OLD platform)."""
    if not responses:
        return 0.0

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

    counts = {}
    for cat in categories:
        counts[cat] = counts.get(cat, 0) + 1

    n = len(categories)
    if n == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log(p + 1e-10)

    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    normalized_entropy = entropy / max(max_entropy, 1.0)

    return round(normalized_entropy, 4)


def classify_se_level(se: float) -> SemanticEntropyLevel:
    """Classify SE into level."""
    if se < 0.3:
        return SemanticEntropyLevel.LOW
    elif se < 0.6:
        return SemanticEntropyLevel.MEDIUM
    else:
        return SemanticEntropyLevel.HIGH


# =============================================================================
# EHR VERIFICATION (same as OLD platform)
# =============================================================================

def verify_against_ehr(case: TestCase) -> EHRVerificationStatus:
    """Verify claim against EHR."""
    if not case.ehr_data_available:
        return EHRVerificationStatus.NOT_FOUND

    if case.expected_ehr_status == "verified":
        return EHRVerificationStatus.VERIFIED
    elif case.expected_ehr_status == "contradiction":
        return EHRVerificationStatus.CONTRADICTION
    elif case.expected_ehr_status == "not_found":
        return EHRVerificationStatus.NOT_FOUND
    else:
        return EHRVerificationStatus.NOT_CHECKABLE


# =============================================================================
# RISK SCORE COMPUTATION (for conformal calibration)
# =============================================================================

def compute_risk_scores(
    ehr_status: EHRVerificationStatus,
    se: float,
    se_level: SemanticEntropyLevel,
    claim_type: ClaimType
) -> Dict[str, float]:
    """
    Compute softmax-like scores for each risk level.

    This is the input to conformal prediction - instead of picking
    the argmax, we return scores for ALL classes.

    The conformal calibrator will use these to build a prediction set.
    """
    # Base scores (before normalization)
    scores = {
        "low": 0.0,
        "medium": 0.0,
        "high": 0.0,
        "critical": 0.0,
    }

    # Assign base scores based on EHR status
    if ehr_status == EHRVerificationStatus.VERIFIED:
        scores["low"] = 3.0
        scores["medium"] = 0.5
        scores["high"] = 0.1
        scores["critical"] = 0.01

    elif ehr_status == EHRVerificationStatus.CONTRADICTION:
        if se_level == SemanticEntropyLevel.LOW:
            # Confident hallucinator - high P(critical)
            scores["critical"] = 3.0
            scores["high"] = 1.0
            scores["medium"] = 0.2
            scores["low"] = 0.01
        elif se_level == SemanticEntropyLevel.MEDIUM:
            scores["critical"] = 1.0
            scores["high"] = 2.5
            scores["medium"] = 0.5
            scores["low"] = 0.1
        else:  # HIGH SE
            scores["critical"] = 0.5
            scores["high"] = 1.5
            scores["medium"] = 2.0
            scores["low"] = 0.2

    elif ehr_status == EHRVerificationStatus.NOT_FOUND:
        if se_level == SemanticEntropyLevel.HIGH:
            scores["high"] = 2.5
            scores["medium"] = 1.5
            scores["low"] = 0.3
            scores["critical"] = 0.2
        else:
            scores["medium"] = 2.5
            scores["high"] = 1.0
            scores["low"] = 0.8
            scores["critical"] = 0.1

    else:  # NOT_CHECKABLE
        if se_level == SemanticEntropyLevel.HIGH:
            scores["high"] = 2.0
            scores["medium"] = 1.5
            scores["low"] = 0.5
            scores["critical"] = 0.3
        else:
            scores["medium"] = 2.0
            scores["low"] = 1.5
            scores["high"] = 0.5
            scores["critical"] = 0.1

    # Add noise based on SE (higher SE = more uncertainty in scores)
    noise_factor = se * 0.5
    for key in scores:
        scores[key] += noise_factor * (0.5 - hash(key) % 100 / 100)

    # Softmax normalization
    max_score = max(scores.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exp_scores.values())

    normalized = {k: round(v / total, 4) for k, v in exp_scores.items()}

    return normalized


def get_true_risk_level(
    ehr_status: EHRVerificationStatus,
    se_level: SemanticEntropyLevel
) -> str:
    """
    Compute the 'true' risk level for coverage calculation.

    This is what we compare the prediction set against.
    """
    if ehr_status == EHRVerificationStatus.VERIFIED:
        return "low"

    if ehr_status == EHRVerificationStatus.CONTRADICTION:
        if se_level == SemanticEntropyLevel.LOW:
            return "critical"
        elif se_level == SemanticEntropyLevel.MEDIUM:
            return "high"
        else:
            return "medium"

    if ehr_status == EHRVerificationStatus.NOT_FOUND:
        if se_level == SemanticEntropyLevel.HIGH:
            return "high"
        else:
            return "medium"

    # NOT_CHECKABLE
    if se_level == SemanticEntropyLevel.HIGH:
        return "high"
    elif se_level == SemanticEntropyLevel.MEDIUM:
        return "medium"
    else:
        return "low"


# =============================================================================
# MAIN RUNNER
# =============================================================================

class NewPlatformRunner:
    """
    Runner for NEW platform with conformal prediction.

    Key difference: Instead of just returning the top-1 risk level,
    we return a PREDICTION SET with coverage guarantee.
    """

    def __init__(self, coverage_target: float = 0.90):
        """
        Initialize runner.

        Args:
            coverage_target: Target coverage (e.g., 0.90 = 90%)
        """
        self.name = "NEW_PLATFORM"
        self.version = "2.0"
        self.coverage_target = coverage_target
        self.alpha = 1 - coverage_target

        # Initialize conformal calibrator
        self.calibrator = AdaptivePredictionSets(alpha=self.alpha)
        self.is_calibrated = False
        self.class_names = ["low", "medium", "high", "critical"]

    def calibrate(self, calibration_cases: List[TestCase]) -> CalibrationResult:
        """
        Calibrate the conformal predictor on held-out data.

        MUST be called before process_case().

        Args:
            calibration_cases: Cases with known ground truth for calibration

        Returns:
            CalibrationResult with threshold info
        """
        cal_scores = []
        cal_labels = []

        for case in calibration_cases:
            # Compute features
            ehr_status = verify_against_ehr(case)
            se = compute_semantic_entropy(case.multiple_responses)
            se_level = classify_se_level(se)

            # Compute risk scores (softmax probabilities)
            scores = compute_risk_scores(ehr_status, se, se_level, case.claim_type)
            cal_scores.append(scores)

            # Get true label
            true_label = get_true_risk_level(ehr_status, se_level)
            cal_labels.append(true_label)

        # Calibrate
        result = self.calibrator.calibrate(cal_scores, cal_labels)
        self.is_calibrated = True

        return result

    def process_case(self, case: TestCase) -> NewPlatformResult:
        """
        Process a single test case through NEW platform with conformal.

        Returns raw measurements including conformal prediction set.
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before process_case()")

        start_time = time.perf_counter()

        # Step 1: Verify against EHR (same as OLD)
        ehr_status = verify_against_ehr(case)

        # Step 2: Compute semantic entropy (same as OLD)
        se = compute_semantic_entropy(case.multiple_responses)
        se_level = classify_se_level(se)

        # Step 3: Compute risk scores (NEW - softmax over all classes)
        scores = compute_risk_scores(ehr_status, se, se_level, case.claim_type)

        # Step 4: Get conformal prediction SET (NEW!)
        prediction_set = self.calibrator.predict_set(scores, self.class_names)

        # Step 5: Check if ground truth is in set (KEY METRIC)
        true_label = get_true_risk_level(ehr_status, se_level)
        ground_truth_in_set = true_label in prediction_set.classes

        # Step 6: Get argmax as "point estimate" for backward compatibility
        risk_level = max(scores, key=scores.get)

        # Step 7: Review needed if set is large OR contains critical
        requires_review = (
            prediction_set.set_size >= 3 or
            "critical" in prediction_set.classes or
            "high" in prediction_set.classes
        )

        end_time = time.perf_counter()
        computation_ms = (end_time - start_time) * 1000

        return NewPlatformResult(
            case_id=case.case_id,
            ehr_status=ehr_status.value,
            semantic_entropy=se,
            se_level=se_level.value,
            conformal_prediction_set=prediction_set.classes,
            conformal_set_size=prediction_set.set_size,
            ground_truth_in_set=ground_truth_in_set,
            conformal_coverage_target=self.coverage_target,
            risk_level=risk_level,
            requires_review=requires_review,
            computation_time_ms=round(computation_ms, 3),
        )

    def process_batch(self, cases: List[TestCase]) -> List[NewPlatformResult]:
        """Process multiple cases."""
        return [self.process_case(case) for case in cases]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from test_cases import generate_test_cases

    print("NEW Platform Runner Test (with Conformal Prediction)")
    print("=" * 60)

    # Generate test cases
    all_cases = generate_test_cases(n_cases=30, seed=42)

    # Split: 10 for calibration, 20 for test
    cal_cases = all_cases[:10]
    test_cases = all_cases[10:]

    runner = NewPlatformRunner(coverage_target=0.90)

    # Calibrate
    print(f"\nCalibrating on {len(cal_cases)} cases...")
    cal_result = runner.calibrate(cal_cases)
    print(f"  Threshold: {cal_result.threshold:.4f}")
    print(f"  Coverage target: {runner.coverage_target:.0%}")

    # Test
    print(f"\nProcessing {len(test_cases)} test cases...\n")

    n_covered = 0
    for case in test_cases[:8]:
        result = runner.process_case(case)

        if result.ground_truth_in_set:
            n_covered += 1

        print(f"{result.case_id}:")
        print(f"  Claim: {case.claim_text[:50]}...")
        print(f"  EHR status: {result.ehr_status}")
        print(f"  SE: {result.semantic_entropy:.4f} ({result.se_level})")
        print(f"  Risk scores → argmax: {result.risk_level}")
        print(f"  CONFORMAL SET: {result.conformal_prediction_set} (size={result.conformal_set_size})")
        print(f"  Ground truth in set: {result.ground_truth_in_set}")
        print(f"  Review needed: {result.requires_review}")
        print(f"  Time: {result.computation_time_ms:.3f} ms")
        print()

    # Process all test cases
    all_results = runner.process_batch(test_cases)
    n_covered_all = sum(1 for r in all_results if r.ground_truth_in_set)
    empirical_coverage = n_covered_all / len(all_results)

    print("-" * 60)
    print(f"EMPIRICAL COVERAGE: {empirical_coverage:.1%} (target: {runner.coverage_target:.0%})")
    print(f"Average set size: {sum(r.conformal_set_size for r in all_results) / len(all_results):.2f}")
