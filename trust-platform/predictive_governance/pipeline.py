"""
Predictive Governance Pipeline
==============================

Implements the Input-Validation-First flow for predictive AI systems.

Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    PredictiveGovernancePipeline                       │
    ├──────────────────────────────────────────────────────────────────────┤
    │  1. VALIDATE: Check input quality and plausibility                   │
    │              ↓                                                        │
    │     ┌─────────────┴─────────────┐                                    │
    │     │                           │                                    │
    │  VALID ✓                    INVALID ✗                                │
    │  (Proceed)                  (Reject/Flag)                            │
    │     │                           │                                    │
    │     ↓                           ↓                                    │
    │  2. PREDICT: Run model       Return error                            │
    │              ↓                                                        │
    │  3. CALIBRATE: Apply conformal prediction for interval               │
    │              ↓                                                        │
    │  4. OUTPUT: Calibrated risk interval + confidence                    │
    │              ↓                                                        │
    │  5. TRACK: Record for outcome tracking (optional)                    │
    └──────────────────────────────────────────────────────────────────────┘

Why Input Validation First?
    Predictive models assume inputs are valid. Given implausible inputs,
    they will still produce a prediction - but it will be meaningless.

    Example: A sepsis risk model expects:
    - Heart rate: 40-200 bpm
    - Temperature: 32-42°C
    - WBC: 1-40 K/uL

    If any input is outside these ranges (or missing), the model prediction
    is unreliable. Better to reject early than propagate garbage.

Example:
    >>> from predictive_governance import PredictiveGovernancePipeline
    >>> from adapters.clinical_risk import ClinicalRiskAdapter
    >>>
    >>> pipeline = PredictiveGovernancePipeline(adapter=ClinicalRiskAdapter())
    >>> result = await pipeline.analyze(
    ...     model_input={"heart_rate": 95, "temperature": 38.5, "wbc": 15.2},
    ...     model_output={"risk_score": 0.35}
    ... )
    >>> print(f"Risk interval: [{result.prediction.lower:.2f}, {result.prediction.upper:.2f}]")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from core_engine.conformal_calibrator import (
    ConformalCalibrator,
    ConformizedQuantileRegression,
    PredictionInterval,
    CalibrationResult,
)
from core_engine.drift_monitor import (
    DriftMonitor,
    DriftResult,
    FeatureDriftDetector,
)


__all__ = [
    'PredictiveGovernancePipeline',
    'PredictiveGovernanceResult',
    'CalibratedPrediction',
    'PredictiveRiskLevel',
]


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class PredictiveRiskLevel(str, Enum):
    """Risk level for predictive AI output."""
    RELIABLE = "reliable"          # Valid inputs, good calibration
    MODERATE = "moderate"          # Minor input issues or wide interval
    UNRELIABLE = "unreliable"      # Input issues or calibration concerns
    REJECTED = "rejected"          # Invalid inputs, prediction rejected


class ActionLevel(str, Enum):
    """Recommended action based on prediction."""
    ROUTINE = "routine"            # Normal monitoring
    ELEVATED = "elevated"          # Increased monitoring
    URGENT = "urgent"              # Prompt clinical review
    CRITICAL = "critical"          # Immediate intervention


@dataclass
class CalibratedPrediction:
    """
    A calibrated prediction with uncertainty interval.

    Instead of just "risk = 0.35", provides "[0.28, 0.42] with 90% coverage"
    """
    point_estimate: float
    lower: float
    upper: float
    coverage_guarantee: float
    interval_width: float
    is_narrow: bool  # True if interval is clinically useful

    @property
    def midpoint(self) -> float:
        """Midpoint of the interval."""
        return (self.lower + self.upper) / 2

    def contains(self, value: float) -> bool:
        """Check if value is within the interval."""
        return self.lower <= value <= self.upper


@dataclass
class InputValidationSummary:
    """Summary of input validation results."""
    n_inputs: int
    n_valid: int
    n_invalid: int
    n_missing: int
    is_acceptable: bool
    critical_issues: List[str]


@dataclass
class PredictiveGovernanceResult:
    """
    Complete result from predictive governance pipeline.

    Contains:
    - Input validation summary
    - Calibrated prediction with uncertainty
    - Recommended action level
    - Drift monitoring status
    """
    risk_level: PredictiveRiskLevel
    action_level: ActionLevel
    prediction: Optional[CalibratedPrediction]
    input_validation: InputValidationSummary
    drift_detected: bool
    recommended_action: str
    tracking_id: Optional[str]  # ID for outcome tracking
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True if prediction is valid and can be used."""
        return self.risk_level != PredictiveRiskLevel.REJECTED

    @property
    def needs_review(self) -> bool:
        """True if prediction needs clinical review."""
        return self.action_level in (ActionLevel.URGENT, ActionLevel.CRITICAL)


# =============================================================================
# ADAPTER PROTOCOL
# =============================================================================

class PredictiveAdapter(Protocol):
    """Protocol for domain-specific predictive adapters."""

    def validate_input(
        self,
        input_name: str,
        value: Any
    ) -> Tuple[bool, Optional[str]]:
        """Validate a single input value."""
        ...

    def get_required_inputs(self) -> List[str]:
        """Get list of required input names."""
        ...

    def get_action_thresholds(self) -> Dict[str, float]:
        """Get thresholds for action levels."""
        ...


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class PredictiveGovernancePipeline:
    """
    Input-Validation-First governance pipeline for predictive AI systems.

    This pipeline ensures:
    1. Inputs are validated before model inference
    2. Predictions are calibrated with uncertainty intervals
    3. Outcomes are tracked for continuous improvement

    Key Features:
    - Early rejection of invalid inputs
    - Conformal prediction for guaranteed coverage
    - Continuous drift monitoring
    - Outcome tracking for calibration
    """

    def __init__(
        self,
        adapter: PredictiveAdapter,
        calibrator: Optional[ConformizedQuantileRegression] = None,
        coverage_target: float = 0.9,
        reject_on_missing: bool = True,
        max_missing_fraction: float = 0.2,
    ):
        """
        Initialize predictive governance pipeline.

        Args:
            adapter: Domain-specific predictive adapter
            calibrator: Optional CQR calibrator (created if not provided)
            coverage_target: Target coverage level (default 90%)
            reject_on_missing: If True, reject if required inputs missing
            max_missing_fraction: Maximum fraction of inputs that can be missing
        """
        self.adapter = adapter
        self.calibrator = calibrator or ConformizedQuantileRegression(alpha=1 - coverage_target)
        self.coverage_target = coverage_target
        self.reject_on_missing = reject_on_missing
        self.max_missing_fraction = max_missing_fraction

        self.drift_monitor = DriftMonitor()
        self._tracking_counter = 0

    def calibrate(
        self,
        lower_quantiles: List[float],
        upper_quantiles: List[float],
        true_values: List[float],
    ) -> CalibrationResult:
        """
        Calibrate the conformal predictor on held-out data.

        Args:
            lower_quantiles: Model's lower quantile predictions (calibration set)
            upper_quantiles: Model's upper quantile predictions (calibration set)
            true_values: True outcomes for calibration set

        Returns:
            CalibrationResult with threshold and statistics
        """
        return self.calibrator.calibrate(lower_quantiles, upper_quantiles, true_values)

    async def analyze(
        self,
        model_input: Dict[str, Any],
        model_output: Dict[str, float],
        quantile_predictions: Optional[Tuple[float, float]] = None,
        track_outcome: bool = True,
    ) -> PredictiveGovernanceResult:
        """
        Analyze model prediction using Input-Validation-First flow.

        Args:
            model_input: Dictionary of input features
            model_output: Model's output (must contain "risk_score" or similar)
            quantile_predictions: Optional (lower, upper) quantile predictions
            track_outcome: If True, generate tracking ID for outcome follow-up

        Returns:
            PredictiveGovernanceResult with calibrated prediction
        """
        # Step 1: Validate inputs
        validation_result = self._validate_inputs(model_input)

        # Step 2: Check if we should reject
        if not validation_result.is_acceptable:
            return self._create_rejected_result(validation_result)

        # Step 3: Get point estimate
        point_estimate = self._extract_point_estimate(model_output)

        # Step 4: Generate calibrated prediction interval
        if quantile_predictions:
            lower_q, upper_q = quantile_predictions
        else:
            # If no quantiles provided, create synthetic interval around point estimate
            # In production, the model should provide quantile predictions
            margin = 0.1  # 10% margin
            lower_q = point_estimate - margin
            upper_q = point_estimate + margin

        # Apply conformal calibration
        if self.calibrator.is_calibrated:
            interval = self.calibrator.predict(lower_q, upper_q, point_estimate)
            calibrated_prediction = CalibratedPrediction(
                point_estimate=point_estimate,
                lower=interval.lower,
                upper=interval.upper,
                coverage_guarantee=self.coverage_target,
                interval_width=interval.width,
                is_narrow=interval.is_narrow,
            )
        else:
            # Not calibrated - use raw quantiles with warning
            calibrated_prediction = CalibratedPrediction(
                point_estimate=point_estimate,
                lower=lower_q,
                upper=upper_q,
                coverage_guarantee=0.0,  # No guarantee
                interval_width=upper_q - lower_q,
                is_narrow=(upper_q - lower_q) < 0.2,
            )

        # Step 5: Check for drift
        drift_result = self._check_drift(model_input)

        # Step 6: Determine risk and action levels
        risk_level = self._compute_risk_level(
            validation_result, calibrated_prediction, drift_result
        )
        action_level = self._compute_action_level(
            calibrated_prediction, self.adapter.get_action_thresholds()
        )

        # Step 7: Generate tracking ID if requested
        tracking_id = None
        if track_outcome:
            self._tracking_counter += 1
            tracking_id = f"PRED-{self._tracking_counter:06d}"

        recommended_action = self._get_recommended_action(
            risk_level, action_level, calibrated_prediction
        )

        return PredictiveGovernanceResult(
            risk_level=risk_level,
            action_level=action_level,
            prediction=calibrated_prediction,
            input_validation=validation_result,
            drift_detected=drift_result.is_significant if drift_result else False,
            recommended_action=recommended_action,
            tracking_id=tracking_id,
            metadata={
                "calibrator_calibrated": self.calibrator.is_calibrated,
                "coverage_target": self.coverage_target,
            }
        )

    def _validate_inputs(self, model_input: Dict[str, Any]) -> InputValidationSummary:
        """Validate all model inputs."""
        required_inputs = self.adapter.get_required_inputs()

        n_valid = 0
        n_invalid = 0
        n_missing = 0
        critical_issues: List[str] = []

        for input_name in required_inputs:
            if input_name not in model_input or model_input[input_name] is None:
                n_missing += 1
                if self.reject_on_missing:
                    critical_issues.append(f"Missing required input: {input_name}")
            else:
                is_valid, error_msg = self.adapter.validate_input(
                    input_name, model_input[input_name]
                )
                if is_valid:
                    n_valid += 1
                else:
                    n_invalid += 1
                    if error_msg:
                        critical_issues.append(error_msg)

        # Check additional provided inputs
        for input_name, value in model_input.items():
            if input_name not in required_inputs and value is not None:
                is_valid, error_msg = self.adapter.validate_input(input_name, value)
                if is_valid:
                    n_valid += 1
                else:
                    n_invalid += 1
                    if error_msg:
                        critical_issues.append(error_msg)

        n_total = n_valid + n_invalid + n_missing
        missing_fraction = n_missing / len(required_inputs) if required_inputs else 0

        is_acceptable = (
            n_invalid == 0 and
            missing_fraction <= self.max_missing_fraction and
            len(critical_issues) == 0
        )

        return InputValidationSummary(
            n_inputs=n_total,
            n_valid=n_valid,
            n_invalid=n_invalid,
            n_missing=n_missing,
            is_acceptable=is_acceptable,
            critical_issues=critical_issues,
        )

    def _extract_point_estimate(self, model_output: Dict[str, float]) -> float:
        """Extract the main point estimate from model output."""
        # Try common keys
        for key in ["risk_score", "probability", "prediction", "score", "risk"]:
            if key in model_output:
                return model_output[key]

        # Fall back to first value
        if model_output:
            return next(iter(model_output.values()))

        return 0.5  # Default

    def _check_drift(self, model_input: Dict[str, Any]) -> Optional[DriftResult]:
        """Check for input drift."""
        # Convert inputs to format expected by drift monitor
        features = {
            k: [float(v)] for k, v in model_input.items()
            if isinstance(v, (int, float))
        }

        if not features:
            return None

        return self.drift_monitor.check(features)

    def _compute_risk_level(
        self,
        validation: InputValidationSummary,
        prediction: CalibratedPrediction,
        drift_result: Optional[DriftResult],
    ) -> PredictiveRiskLevel:
        """Compute overall risk level for the prediction."""
        if not validation.is_acceptable:
            return PredictiveRiskLevel.REJECTED

        if drift_result and drift_result.is_significant:
            return PredictiveRiskLevel.UNRELIABLE

        if validation.n_missing > 0:
            return PredictiveRiskLevel.MODERATE

        if not prediction.is_narrow:
            return PredictiveRiskLevel.MODERATE

        return PredictiveRiskLevel.RELIABLE

    def _compute_action_level(
        self,
        prediction: CalibratedPrediction,
        thresholds: Dict[str, float],
    ) -> ActionLevel:
        """Compute recommended action level based on prediction."""
        # Use the upper bound of the interval for safety
        risk_value = prediction.upper

        critical_threshold = thresholds.get("critical", 0.8)
        urgent_threshold = thresholds.get("urgent", 0.5)
        elevated_threshold = thresholds.get("elevated", 0.3)

        if risk_value >= critical_threshold:
            return ActionLevel.CRITICAL
        elif risk_value >= urgent_threshold:
            return ActionLevel.URGENT
        elif risk_value >= elevated_threshold:
            return ActionLevel.ELEVATED
        else:
            return ActionLevel.ROUTINE

    def _create_rejected_result(
        self,
        validation: InputValidationSummary,
    ) -> PredictiveGovernanceResult:
        """Create a rejected result for invalid inputs."""
        return PredictiveGovernanceResult(
            risk_level=PredictiveRiskLevel.REJECTED,
            action_level=ActionLevel.ROUTINE,
            prediction=None,
            input_validation=validation,
            drift_detected=False,
            recommended_action=f"REJECTED: {'; '.join(validation.critical_issues)}",
            tracking_id=None,
        )

    def _get_recommended_action(
        self,
        risk_level: PredictiveRiskLevel,
        action_level: ActionLevel,
        prediction: CalibratedPrediction,
    ) -> str:
        """Get human-readable recommended action."""
        if risk_level == PredictiveRiskLevel.REJECTED:
            return "Prediction rejected due to input validation failures."

        interval_str = f"[{prediction.lower:.2f}, {prediction.upper:.2f}]"

        if action_level == ActionLevel.CRITICAL:
            return f"CRITICAL: Risk interval {interval_str}. Immediate clinical review required."
        elif action_level == ActionLevel.URGENT:
            return f"URGENT: Risk interval {interval_str}. Prompt clinical assessment needed."
        elif action_level == ActionLevel.ELEVATED:
            return f"ELEVATED: Risk interval {interval_str}. Increased monitoring recommended."
        else:
            return f"ROUTINE: Risk interval {interval_str}. Continue standard monitoring."
