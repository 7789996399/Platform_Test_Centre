"""
Diagnostic Governance Pipeline
==============================

Implements the Conformal-First verification flow for diagnostic AI systems.

Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    DiagnosticGovernancePipeline                       │
    ├──────────────────────────────────────────────────────────────────────┤
    │  1. CONFORMAL: Generate prediction SET with coverage guarantee       │
    │              ↓                                                        │
    │  2. CONTEXT: Integrate prior imaging and clinical history            │
    │              ↓                                                        │
    │  3. VALIDATE: Check attention maps for plausibility                  │
    │              ↓                                                        │
    │     ┌─────────────┴─────────────┐                                    │
    │     │                           │                                    │
    │  SINGLETON SET              LARGE SET                                │
    │  (High confidence)          (Uncertain)                              │
    │     │                           │                                    │
    │     │                    Check attention                             │
    │     │                    Check priors                                │
    │     └─────────────┬─────────────┘                                    │
    │              ↓                                                        │
    │  4. OUTPUT: Calibrated prediction set + context + review guidance    │
    └──────────────────────────────────────────────────────────────────────┘

Why Conformal-First for Diagnostics?
    Unlike generative AI where we verify claims against sources, diagnostic AI
    produces classifications. The key question is: "How uncertain is this prediction?"

    Conformal prediction answers this directly:
    - Small prediction set → Model is confident
    - Large prediction set → Model is uncertain
    - Coverage guarantee → Mathematical bound on error rate

Example:
    >>> from diagnostic_governance import DiagnosticGovernancePipeline
    >>> from adapters.radiology import RadiologyAdapter
    >>>
    >>> pipeline = DiagnosticGovernancePipeline(adapter=RadiologyAdapter())
    >>> result = await pipeline.analyze(
    ...     model_scores={"normal": 0.7, "pneumonia": 0.2, "atelectasis": 0.1},
    ...     prior_studies=[previous_xray],
    ...     image_data=current_xray
    ... )
    >>> print(f"Prediction set: {result.prediction.classes}")
    >>> print(f"Coverage guarantee: {result.coverage_guarantee}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

from core_engine.conformal_calibrator import (
    ConformalCalibrator,
    PredictionSet,
    CalibrationResult,
)
from core_engine.drift_monitor import (
    CoverageMonitor,
    DriftResult,
)


__all__ = [
    'DiagnosticGovernancePipeline',
    'DiagnosticGovernanceResult',
    'DiagnosticPrediction',
    'DiagnosticRiskLevel',
]


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class DiagnosticRiskLevel(str, Enum):
    """Risk level for diagnostic AI output."""
    LOW = "low"                    # Singleton prediction set, good attention
    MODERATE = "moderate"          # Small set or minor attention concerns
    HIGH = "high"                  # Large set or attention issues
    UNCERTAIN = "uncertain"        # Very large set, defer to human
    INVALID = "invalid"            # Attention validation failed


class DiagnosticReviewLevel(str, Enum):
    """Review level for diagnostic findings."""
    NONE = "none"                  # AI result can be used directly
    QUICK_CHECK = "quick_check"    # 30 second verification
    STANDARD = "standard"          # Normal radiologist review
    DETAILED = "detailed"          # Careful review with priors
    REJECT = "reject"              # AI result should not be used


@dataclass
class DiagnosticPrediction:
    """
    A diagnostic prediction with conformal calibration.

    Contains the prediction set (not just top-1), coverage guarantee,
    and confidence assessment.
    """
    prediction_set: PredictionSet
    top_prediction: str
    top_score: float
    coverage_guarantee: float  # e.g., 0.9 = 90% coverage
    set_size: int
    is_confident: bool
    raw_scores: Dict[str, float]

    @property
    def is_singleton(self) -> bool:
        """True if prediction set contains exactly one class."""
        return self.set_size == 1

    @property
    def is_uncertain(self) -> bool:
        """True if prediction set is large (>3 classes typically)."""
        return self.set_size > 3


@dataclass
class ContextualModification:
    """Modification to prediction based on context."""
    original_prediction: str
    modified_prediction: Optional[str]
    reason: str
    confidence_adjustment: float  # Positive = more confident, negative = less


@dataclass
class DiagnosticGovernanceResult:
    """
    Complete result from diagnostic governance pipeline.

    Contains:
    - Calibrated prediction with coverage guarantee
    - Prior study integration results
    - Attention validation results
    - Recommended review level
    - Drift monitoring status
    """
    risk_level: DiagnosticRiskLevel
    review_level: DiagnosticReviewLevel
    prediction: DiagnosticPrediction
    contextual_modifications: List[ContextualModification]
    attention_valid: bool
    attention_concerns: List[str]
    prior_comparison: Optional[str]
    drift_detected: bool
    recommended_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """True if AI result can be acted upon (with appropriate review)."""
        return self.review_level != DiagnosticReviewLevel.REJECT

    @property
    def needs_human_review(self) -> bool:
        """True if human review is required."""
        return self.review_level not in (DiagnosticReviewLevel.NONE, DiagnosticReviewLevel.QUICK_CHECK)


# =============================================================================
# ADAPTER PROTOCOL
# =============================================================================

class DiagnosticAdapter(Protocol):
    """Protocol for domain-specific diagnostic adapters."""

    def get_class_names(self) -> List[str]:
        """Get list of possible diagnostic classes."""
        ...

    def validate_attention(
        self,
        attention_map: Any,
        predicted_class: str
    ) -> Tuple[bool, List[str]]:
        """Validate attention map for predicted class."""
        ...

    def compare_with_prior(
        self,
        current_prediction: str,
        prior_predictions: List[str]
    ) -> str:
        """Compare current finding with prior studies."""
        ...


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class DiagnosticGovernancePipeline:
    """
    Conformal-First governance pipeline for diagnostic AI systems.

    This pipeline transforms uncertain point predictions into calibrated
    prediction sets with coverage guarantees. It integrates prior context
    and validates attention maps to catch model failures.

    Key Features:
    - Conformal prediction for coverage guarantees
    - Prior study integration for context
    - Attention map validation
    - Continuous coverage monitoring for drift detection
    """

    def __init__(
        self,
        adapter: DiagnosticAdapter,
        calibrator: Optional[ConformalCalibrator] = None,
        coverage_target: float = 0.9,
        max_set_size_for_confidence: int = 2,
    ):
        """
        Initialize diagnostic governance pipeline.

        Args:
            adapter: Domain-specific diagnostic adapter
            calibrator: Optional conformal calibrator (created if not provided)
            coverage_target: Target coverage level (default 90%)
            max_set_size_for_confidence: Max set size to consider "confident"
        """
        self.adapter = adapter
        self.calibrator = calibrator or ConformalCalibrator(alpha=1 - coverage_target)
        self.coverage_target = coverage_target
        self.max_set_size_for_confidence = max_set_size_for_confidence
        self.coverage_monitor = CoverageMonitor(
            target_coverage=coverage_target,
            window_size=100
        )

    def calibrate(
        self,
        calibration_scores: List[Dict[str, float]],
        calibration_labels: List[str],
    ) -> CalibrationResult:
        """
        Calibrate the conformal predictor on held-out data.

        Must be called before analyze() to enable conformal prediction.

        Args:
            calibration_scores: Model softmax scores for calibration set
            calibration_labels: True labels for calibration set

        Returns:
            CalibrationResult with threshold and statistics
        """
        return self.calibrator.calibrate(calibration_scores, calibration_labels)

    async def analyze(
        self,
        model_scores: Dict[str, float],
        prior_studies: Optional[List[Any]] = None,
        attention_map: Optional[Any] = None,
        clinical_context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticGovernanceResult:
        """
        Analyze diagnostic model output using Conformal-First flow.

        Args:
            model_scores: Model's softmax/probability scores per class
            prior_studies: Optional list of prior study results
            attention_map: Optional attention/saliency map
            clinical_context: Optional clinical context (history, symptoms)

        Returns:
            DiagnosticGovernanceResult with calibrated prediction and guidance
        """
        # Step 1: Generate conformal prediction set
        class_names = self.adapter.get_class_names()
        prediction_set = self.calibrator.predict_set(model_scores, class_names)

        # Create diagnostic prediction
        top_class = max(model_scores.keys(), key=lambda k: model_scores[k])
        top_score = model_scores[top_class]

        prediction = DiagnosticPrediction(
            prediction_set=prediction_set,
            top_prediction=top_class,
            top_score=top_score,
            coverage_guarantee=self.coverage_target,
            set_size=prediction_set.set_size,
            is_confident=prediction_set.set_size <= self.max_set_size_for_confidence,
            raw_scores=model_scores,
        )

        # Step 2: Validate attention map
        attention_valid = True
        attention_concerns: List[str] = []
        if attention_map is not None:
            attention_valid, attention_concerns = self.adapter.validate_attention(
                attention_map, top_class
            )

        # Step 3: Compare with prior studies
        prior_comparison = None
        contextual_modifications: List[ContextualModification] = []
        if prior_studies:
            prior_predictions = [str(p) for p in prior_studies]  # Simplified
            prior_comparison = self.adapter.compare_with_prior(
                top_class, prior_predictions
            )

            # Check for concerning changes
            if prior_comparison and "new finding" in prior_comparison.lower():
                contextual_modifications.append(ContextualModification(
                    original_prediction=top_class,
                    modified_prediction=None,
                    reason="New finding compared to prior - requires careful review",
                    confidence_adjustment=-0.1,
                ))

        # Step 4: Check for drift
        drift_result = self.coverage_monitor.check()
        drift_detected = drift_result.is_significant

        # Step 5: Compute risk and review levels
        risk_level = self._compute_risk_level(
            prediction, attention_valid, attention_concerns, drift_detected
        )
        review_level = self._compute_review_level(
            risk_level, prediction, contextual_modifications
        )
        recommended_action = self._get_recommended_action(
            risk_level, review_level, prediction
        )

        return DiagnosticGovernanceResult(
            risk_level=risk_level,
            review_level=review_level,
            prediction=prediction,
            contextual_modifications=contextual_modifications,
            attention_valid=attention_valid,
            attention_concerns=attention_concerns,
            prior_comparison=prior_comparison,
            drift_detected=drift_detected,
            recommended_action=recommended_action,
            metadata={
                "calibrator_threshold": self.calibrator._threshold,
                "coverage_monitor_coverage": self.coverage_monitor.current_coverage,
            }
        )

    def update_coverage(self, true_label: str, prediction_set: PredictionSet) -> None:
        """
        Update coverage monitor with a verified prediction.

        Call this when ground truth becomes available to track coverage.

        Args:
            true_label: The actual true label
            prediction_set: The prediction set that was generated
        """
        is_covered = true_label in prediction_set.classes
        self.coverage_monitor.update(is_covered)

    def _compute_risk_level(
        self,
        prediction: DiagnosticPrediction,
        attention_valid: bool,
        attention_concerns: List[str],
        drift_detected: bool,
    ) -> DiagnosticRiskLevel:
        """Compute risk level from prediction characteristics."""
        # Invalid attention = reject
        if not attention_valid and len(attention_concerns) >= 3:
            return DiagnosticRiskLevel.INVALID

        # Drift = uncertain
        if drift_detected:
            return DiagnosticRiskLevel.UNCERTAIN

        # Large prediction set = uncertain
        if prediction.set_size > 5:
            return DiagnosticRiskLevel.UNCERTAIN

        # Some attention concerns = high
        if attention_concerns:
            return DiagnosticRiskLevel.HIGH

        # Moderate set size = moderate
        if prediction.set_size > 2:
            return DiagnosticRiskLevel.MODERATE

        # Singleton or small set = low
        return DiagnosticRiskLevel.LOW

    def _compute_review_level(
        self,
        risk_level: DiagnosticRiskLevel,
        prediction: DiagnosticPrediction,
        contextual_modifications: List[ContextualModification],
    ) -> DiagnosticReviewLevel:
        """Compute recommended review level."""
        if risk_level == DiagnosticRiskLevel.INVALID:
            return DiagnosticReviewLevel.REJECT

        if risk_level == DiagnosticRiskLevel.UNCERTAIN:
            return DiagnosticReviewLevel.DETAILED

        if risk_level == DiagnosticRiskLevel.HIGH:
            return DiagnosticReviewLevel.STANDARD

        if contextual_modifications:
            return DiagnosticReviewLevel.STANDARD

        if risk_level == DiagnosticRiskLevel.MODERATE:
            return DiagnosticReviewLevel.QUICK_CHECK

        # Low risk, singleton prediction
        if prediction.is_singleton and prediction.top_score > 0.9:
            return DiagnosticReviewLevel.NONE

        return DiagnosticReviewLevel.QUICK_CHECK

    def _get_recommended_action(
        self,
        risk_level: DiagnosticRiskLevel,
        review_level: DiagnosticReviewLevel,
        prediction: DiagnosticPrediction,
    ) -> str:
        """Get human-readable recommended action."""
        if review_level == DiagnosticReviewLevel.REJECT:
            return "REJECT: AI prediction should not be used. Full manual review required."

        if review_level == DiagnosticReviewLevel.DETAILED:
            return f"DETAILED REVIEW: Prediction set has {prediction.set_size} possibilities. Careful review with priors required."

        if review_level == DiagnosticReviewLevel.STANDARD:
            return f"STANDARD REVIEW: Normal radiologist review recommended for {prediction.top_prediction}."

        if review_level == DiagnosticReviewLevel.QUICK_CHECK:
            return f"QUICK CHECK: Brief verification of {prediction.top_prediction} ({prediction.top_score:.0%} confidence)."

        return f"AI CONFIDENT: {prediction.top_prediction} ({prediction.coverage_guarantee:.0%} coverage guarantee)."
