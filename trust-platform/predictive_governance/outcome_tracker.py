"""
Outcome Tracker
===============

Tracks predictions and outcomes for continuous calibration assessment.

Why Track Outcomes?
    Calibration guarantees are only valid if the model's performance
    matches the calibration data. Over time:
    - Patient populations change
    - Clinical practices evolve
    - Data collection methods change

    By tracking predictions and outcomes, we can:
    1. Detect when calibration is degrading
    2. Trigger recalibration when needed
    3. Provide transparency about actual performance

Tracking Flow:
    1. Record prediction with unique ID
    2. Later, record actual outcome
    3. Periodically compute calibration statistics
    4. Alert if coverage drops below target

Example:
    >>> tracker = OutcomeTracker(coverage_target=0.9)
    >>>
    >>> # At prediction time
    >>> pred_id = tracker.record_prediction(
    ...     prediction_interval=(0.2, 0.4),
    ...     point_estimate=0.3,
    ...     inputs={"heart_rate": 85}
    ... )
    >>>
    >>> # When outcome known
    >>> tracker.record_outcome(pred_id, actual_value=0.35)
    >>>
    >>> # Check calibration
    >>> stats = tracker.get_calibration_stats()
    >>> print(f"Coverage: {stats.empirical_coverage:.1%}")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import uuid


__all__ = [
    'OutcomeTracker',
    'TrackedPrediction',
    'CalibrationStats',
    'CalibrationAlert',
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrackedPrediction:
    """
    A prediction being tracked for outcome.

    Contains the prediction interval, inputs, and eventual outcome.
    """
    prediction_id: str
    timestamp: str  # ISO format
    lower_bound: float
    upper_bound: float
    point_estimate: float
    inputs: Dict[str, Any]
    outcome: Optional[float] = None
    outcome_timestamp: Optional[str] = None
    is_covered: Optional[bool] = None  # True if outcome in interval

    @property
    def has_outcome(self) -> bool:
        """True if outcome has been recorded."""
        return self.outcome is not None

    @property
    def interval_width(self) -> float:
        """Width of the prediction interval."""
        return self.upper_bound - self.lower_bound


@dataclass
class CalibrationStats:
    """Statistics about calibration performance."""
    n_predictions: int
    n_with_outcomes: int
    n_covered: int
    target_coverage: float
    empirical_coverage: float
    coverage_gap: float  # Target - empirical
    is_well_calibrated: bool
    mean_interval_width: float
    recent_coverage: float  # Coverage in recent window


@dataclass
class CalibrationAlert:
    """Alert generated when calibration degrades."""
    alert_type: str  # "coverage_drop", "drift_detected"
    severity: str  # "warning", "critical"
    message: str
    current_coverage: float
    target_coverage: float
    recommendation: str
    timestamp: str


# =============================================================================
# MAIN TRACKER CLASS
# =============================================================================

class OutcomeTracker:
    """
    Tracks predictions and outcomes for calibration assessment.

    Maintains a history of predictions and their outcomes, computes
    calibration statistics, and alerts when performance degrades.

    Example:
        >>> tracker = OutcomeTracker(coverage_target=0.9)
        >>>
        >>> # Record predictions
        >>> for patient in patients:
        ...     pred_id = tracker.record_prediction(
        ...         prediction_interval=(lower, upper),
        ...         point_estimate=risk_score,
        ...         inputs=patient_data
        ...     )
        ...     # Store pred_id for later outcome recording
        >>>
        >>> # Record outcomes as they become available
        >>> tracker.record_outcome("PRED-000001", actual_risk=0.45)
        >>>
        >>> # Monitor calibration
        >>> if tracker.check_calibration_alert():
        ...     print("Calibration degraded - consider recalibration")
    """

    def __init__(
        self,
        coverage_target: float = 0.9,
        window_size: int = 100,
        alert_threshold: float = 0.05,
        max_predictions: int = 10000,
    ):
        """
        Initialize outcome tracker.

        Args:
            coverage_target: Target coverage level (e.g., 0.9 for 90%)
            window_size: Size of rolling window for recent coverage
            alert_threshold: Coverage drop to trigger alert
            max_predictions: Maximum predictions to store
        """
        self.coverage_target = coverage_target
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.max_predictions = max_predictions

        self._predictions: Dict[str, TrackedPrediction] = {}
        self._prediction_order: deque = deque(maxlen=max_predictions)
        self._recent_outcomes: deque = deque(maxlen=window_size)
        self._alerts: List[CalibrationAlert] = []
        self._counter = 0

    def record_prediction(
        self,
        prediction_interval: Tuple[float, float],
        point_estimate: float,
        inputs: Dict[str, Any],
        prediction_id: Optional[str] = None,
    ) -> str:
        """
        Record a prediction for tracking.

        Args:
            prediction_interval: (lower, upper) bounds of prediction
            point_estimate: Point estimate (if any)
            inputs: Model inputs that generated this prediction
            prediction_id: Optional custom ID (generated if not provided)

        Returns:
            prediction_id for later outcome recording
        """
        if prediction_id is None:
            self._counter += 1
            prediction_id = f"PRED-{self._counter:06d}"

        lower, upper = prediction_interval

        prediction = TrackedPrediction(
            prediction_id=prediction_id,
            timestamp=datetime.now().isoformat(),
            lower_bound=lower,
            upper_bound=upper,
            point_estimate=point_estimate,
            inputs=inputs,
        )

        self._predictions[prediction_id] = prediction
        self._prediction_order.append(prediction_id)

        # Remove oldest if at capacity
        while len(self._predictions) > self.max_predictions:
            oldest_id = self._prediction_order.popleft()
            if oldest_id in self._predictions:
                del self._predictions[oldest_id]

        return prediction_id

    def record_outcome(
        self,
        prediction_id: str,
        actual_value: float,
    ) -> bool:
        """
        Record the actual outcome for a prediction.

        Args:
            prediction_id: ID of the prediction
            actual_value: The actual outcome value

        Returns:
            True if outcome was recorded, False if prediction not found
        """
        if prediction_id not in self._predictions:
            return False

        prediction = self._predictions[prediction_id]
        prediction.outcome = actual_value
        prediction.outcome_timestamp = datetime.now().isoformat()
        prediction.is_covered = (
            prediction.lower_bound <= actual_value <= prediction.upper_bound
        )

        # Update recent outcomes
        self._recent_outcomes.append(1 if prediction.is_covered else 0)

        # Check for calibration alert
        self._check_and_create_alert()

        return True

    def get_prediction(self, prediction_id: str) -> Optional[TrackedPrediction]:
        """Get a tracked prediction by ID."""
        return self._predictions.get(prediction_id)

    def get_calibration_stats(self) -> CalibrationStats:
        """
        Compute current calibration statistics.

        Returns:
            CalibrationStats with coverage and interval statistics
        """
        predictions_with_outcomes = [
            p for p in self._predictions.values()
            if p.has_outcome
        ]

        n_with_outcomes = len(predictions_with_outcomes)
        n_covered = sum(1 for p in predictions_with_outcomes if p.is_covered)

        if n_with_outcomes > 0:
            empirical_coverage = n_covered / n_with_outcomes
            mean_width = sum(
                p.interval_width for p in predictions_with_outcomes
            ) / n_with_outcomes
        else:
            empirical_coverage = 1.0
            mean_width = 0.0

        # Recent coverage from rolling window
        if self._recent_outcomes:
            recent_coverage = sum(self._recent_outcomes) / len(self._recent_outcomes)
        else:
            recent_coverage = 1.0

        coverage_gap = self.coverage_target - empirical_coverage
        is_well_calibrated = empirical_coverage >= self.coverage_target - self.alert_threshold

        return CalibrationStats(
            n_predictions=len(self._predictions),
            n_with_outcomes=n_with_outcomes,
            n_covered=n_covered,
            target_coverage=self.coverage_target,
            empirical_coverage=empirical_coverage,
            coverage_gap=coverage_gap,
            is_well_calibrated=is_well_calibrated,
            mean_interval_width=mean_width,
            recent_coverage=recent_coverage,
        )

    def check_calibration_alert(self) -> Optional[CalibrationAlert]:
        """
        Check if calibration has degraded and return alert if so.

        Returns:
            CalibrationAlert if coverage has dropped, None otherwise
        """
        stats = self.get_calibration_stats()

        if stats.n_with_outcomes < self.window_size // 2:
            return None  # Not enough data

        if stats.is_well_calibrated:
            return None

        # Create alert
        if stats.coverage_gap > self.alert_threshold * 2:
            severity = "critical"
            recommendation = "URGENT: Recalibrate model immediately"
        else:
            severity = "warning"
            recommendation = "Consider recalibration within 1 week"

        return CalibrationAlert(
            alert_type="coverage_drop",
            severity=severity,
            message=f"Coverage dropped to {stats.empirical_coverage:.1%} "
                    f"(target: {stats.target_coverage:.1%})",
            current_coverage=stats.empirical_coverage,
            target_coverage=stats.target_coverage,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat(),
        )

    def _check_and_create_alert(self) -> None:
        """Check for alert condition and store if triggered."""
        alert = self.check_calibration_alert()
        if alert:
            self._alerts.append(alert)

    @property
    def alerts(self) -> List[CalibrationAlert]:
        """Get all generated alerts."""
        return self._alerts.copy()

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self._alerts.clear()

    def get_predictions_for_recalibration(
        self,
        n_samples: Optional[int] = None
    ) -> List[TrackedPrediction]:
        """
        Get predictions with outcomes for recalibration.

        Args:
            n_samples: Number of samples to return (None = all)

        Returns:
            List of predictions with recorded outcomes
        """
        predictions_with_outcomes = [
            p for p in self._predictions.values()
            if p.has_outcome
        ]

        if n_samples is not None and len(predictions_with_outcomes) > n_samples:
            # Return most recent
            predictions_with_outcomes.sort(
                key=lambda p: p.outcome_timestamp or "",
                reverse=True
            )
            return predictions_with_outcomes[:n_samples]

        return predictions_with_outcomes

    def export_for_analysis(self) -> Dict[str, Any]:
        """
        Export tracking data for analysis.

        Returns:
            Dictionary with predictions, outcomes, and stats
        """
        return {
            "predictions": [
                {
                    "id": p.prediction_id,
                    "timestamp": p.timestamp,
                    "lower": p.lower_bound,
                    "upper": p.upper_bound,
                    "point_estimate": p.point_estimate,
                    "outcome": p.outcome,
                    "is_covered": p.is_covered,
                }
                for p in self._predictions.values()
            ],
            "stats": {
                "target_coverage": self.coverage_target,
                "n_predictions": len(self._predictions),
            },
            "alerts": [
                {
                    "type": a.alert_type,
                    "severity": a.severity,
                    "coverage": a.current_coverage,
                    "timestamp": a.timestamp,
                }
                for a in self._alerts
            ],
        }
