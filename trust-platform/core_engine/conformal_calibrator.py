"""
TRUST Platform - Conformal Calibrator Module
==============================================

Provides distribution-free uncertainty quantification using conformal prediction.

Why Conformal Prediction?
    Traditional ML outputs point predictions without valid uncertainty estimates.
    Conformal prediction provides mathematically guaranteed coverage:
    "The true value will be in the prediction set with probability >= 1 - alpha"

    This is CRITICAL for healthcare AI where we need to know:
    - How confident is this radiology finding?
    - What's the uncertainty range on this risk score?
    - When should we defer to human review?

Two Main Approaches:

1. Classification: Adaptive Prediction Sets (APS)
   - Returns a SET of possible classes, not just the top-1
   - Set size adapts to model uncertainty
   - Smaller set = more confident, larger set = less confident
   - Example: Radiology AI returns {"normal", "atelectasis"} instead of just "normal"

2. Regression: Conformalized Quantile Regression (CQR)
   - Returns an INTERVAL, not a point estimate
   - Interval width adapts to prediction difficulty
   - Example: Risk score of [0.23, 0.41] instead of just 0.32

Coverage Guarantee:
    Given calibration data and significance level alpha, conformal prediction
    guarantees that P(Y ∈ C(X)) >= 1 - alpha for future data from the same
    distribution. This is a finite-sample guarantee, not an asymptotic one.

Example:
    >>> calibrator = ConformalCalibrator(alpha=0.1)  # 90% coverage
    >>> calibrator.calibrate(cal_scores, cal_labels)
    >>> prediction_set = calibrator.predict_set(new_scores)
    >>> # prediction_set contains true label with >= 90% probability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
import math


__all__ = [
    # Enums
    'CalibrationMethod',
    'CoverageLevel',
    # Data classes
    'CalibrationResult',
    'PredictionSet',
    'PredictionInterval',
    'CoverageStats',
    # Main classes
    'ConformalCalibrator',
    'AdaptivePredictionSets',
    'ConformizedQuantileRegression',
    # Utilities
    'compute_quantile',
    'compute_coverage',
]


# =============================================================================
# ENUMS
# =============================================================================

class CalibrationMethod(str, Enum):
    """Conformal prediction method to use."""
    APS = "adaptive_prediction_sets"      # For classification
    CQR = "conformalized_quantile_regression"  # For regression
    RAPS = "regularized_adaptive_prediction_sets"  # APS with regularization
    LAC = "least_ambiguous_classifer"     # Minimizes set size


class CoverageLevel(str, Enum):
    """Pre-defined coverage levels for common use cases."""
    STANDARD = "0.90"      # 90% coverage - general use
    HIGH = "0.95"          # 95% coverage - higher stakes
    CLINICAL = "0.99"      # 99% coverage - clinical decisions
    SCREENING = "0.999"    # 99.9% coverage - screening (minimize false negatives)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CalibrationResult:
    """Result of calibrating a conformal predictor."""
    method: CalibrationMethod
    alpha: float  # Significance level (1 - coverage)
    threshold: float  # Calibrated threshold/quantile
    n_calibration: int  # Number of calibration samples
    empirical_coverage: float  # Coverage on calibration set
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionSet:
    """
    Prediction set for classification tasks.

    The prediction set contains all classes that cannot be ruled out
    at the given significance level. Smaller sets indicate higher
    model confidence; larger sets indicate uncertainty.
    """
    classes: List[str]  # Classes in the prediction set
    scores: Dict[str, float]  # Softmax/probability scores per class
    threshold: float  # Calibrated threshold used
    alpha: float  # Significance level
    set_size: int  # Number of classes in set

    @property
    def is_singleton(self) -> bool:
        """True if prediction set contains exactly one class (high confidence)."""
        return self.set_size == 1

    @property
    def is_empty(self) -> bool:
        """True if prediction set is empty (should not happen with valid calibration)."""
        return self.set_size == 0

    @property
    def top_class(self) -> Optional[str]:
        """Return the highest-scoring class in the set."""
        if not self.classes:
            return None
        return max(self.classes, key=lambda c: self.scores.get(c, 0))


@dataclass
class PredictionInterval:
    """
    Prediction interval for regression tasks.

    The interval [lower, upper] is guaranteed to contain the true
    value with probability >= 1 - alpha under exchangeability.
    """
    lower: float
    upper: float
    point_estimate: float  # Original model prediction
    alpha: float  # Significance level
    width: float = field(init=False)  # Interval width

    def __post_init__(self):
        self.width = self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Check if value is within the interval."""
        return self.lower <= value <= self.upper

    @property
    def is_narrow(self) -> bool:
        """Heuristic: interval is narrow if width < 0.2 of point estimate."""
        if abs(self.point_estimate) < 1e-6:
            return self.width < 0.1
        return self.width / abs(self.point_estimate) < 0.2


@dataclass
class CoverageStats:
    """Statistics for evaluating coverage guarantees."""
    target_coverage: float  # 1 - alpha
    empirical_coverage: float  # Actual coverage observed
    n_samples: int
    coverage_gap: float = field(init=False)  # Target - empirical
    is_valid: bool = field(init=False)  # True if empirical >= target

    def __post_init__(self):
        self.coverage_gap = self.target_coverage - self.empirical_coverage
        self.is_valid = self.empirical_coverage >= self.target_coverage - 0.01  # 1% tolerance


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_quantile(values: List[float], q: float) -> float:
    """
    Compute quantile with the adjustment for conformal prediction.

    For conformal prediction, we use the (1-alpha)(1 + 1/n) quantile
    of calibration scores to ensure finite-sample coverage guarantee.

    Args:
        values: List of conformity scores
        q: Quantile level (e.g., 0.9 for 90th percentile)

    Returns:
        The q-th quantile of values
    """
    if not values:
        raise ValueError("Cannot compute quantile of empty list")

    sorted_values = sorted(values)
    n = len(sorted_values)

    # Adjusted quantile for finite-sample guarantee
    adjusted_q = min(1.0, q * (1 + 1/n))
    idx = int(math.ceil(adjusted_q * n)) - 1
    idx = max(0, min(idx, n - 1))

    return sorted_values[idx]


def compute_coverage(
    prediction_sets: List[PredictionSet],
    true_labels: List[str]
) -> CoverageStats:
    """
    Compute empirical coverage for classification prediction sets.

    Args:
        prediction_sets: List of prediction sets
        true_labels: Corresponding true labels

    Returns:
        Coverage statistics
    """
    if len(prediction_sets) != len(true_labels):
        raise ValueError("Prediction sets and labels must have same length")

    if not prediction_sets:
        raise ValueError("Cannot compute coverage of empty list")

    covered = sum(
        1 for ps, label in zip(prediction_sets, true_labels)
        if label in ps.classes
    )

    empirical = covered / len(prediction_sets)
    target = 1 - prediction_sets[0].alpha

    return CoverageStats(
        target_coverage=target,
        empirical_coverage=empirical,
        n_samples=len(prediction_sets),
    )


def compute_interval_coverage(
    intervals: List[PredictionInterval],
    true_values: List[float]
) -> CoverageStats:
    """
    Compute empirical coverage for regression prediction intervals.

    Args:
        intervals: List of prediction intervals
        true_values: Corresponding true values

    Returns:
        Coverage statistics
    """
    if len(intervals) != len(true_values):
        raise ValueError("Intervals and values must have same length")

    if not intervals:
        raise ValueError("Cannot compute coverage of empty list")

    covered = sum(
        1 for interval, value in zip(intervals, true_values)
        if interval.contains(value)
    )

    empirical = covered / len(intervals)
    target = 1 - intervals[0].alpha

    return CoverageStats(
        target_coverage=target,
        empirical_coverage=empirical,
        n_samples=len(intervals),
    )


# =============================================================================
# CONFORMAL CALIBRATOR - MAIN CLASS
# =============================================================================

class ConformalCalibrator:
    """
    Main conformal calibrator supporting multiple methods.

    This class provides a unified interface for conformal prediction,
    automatically selecting the appropriate method based on the task
    (classification vs regression).

    Architecture Decision:
        The calibrator is domain-agnostic. Domain-specific logic
        (what scores mean, what classes represent) is handled by
        the governance pipelines and adapters.

    Example (Classification):
        >>> calibrator = ConformalCalibrator(alpha=0.1, method=CalibrationMethod.APS)
        >>> calibrator.calibrate(softmax_scores, labels)
        >>> pred_set = calibrator.predict_set(new_scores)
        >>> print(f"Possible diagnoses: {pred_set.classes}")

    Example (Regression):
        >>> calibrator = ConformalCalibrator(alpha=0.1, method=CalibrationMethod.CQR)
        >>> calibrator.calibrate(quantile_preds, true_values)
        >>> interval = calibrator.predict_interval(new_quantiles)
        >>> print(f"Risk score: [{interval.lower:.2f}, {interval.upper:.2f}]")
    """

    def __init__(
        self,
        alpha: float = 0.1,
        method: CalibrationMethod = CalibrationMethod.APS,
    ):
        """
        Initialize conformal calibrator.

        Args:
            alpha: Significance level (default 0.1 = 90% coverage)
            method: Conformal prediction method to use
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.method = method
        self._calibrated = False
        self._threshold: Optional[float] = None
        self._calibration_result: Optional[CalibrationResult] = None

    @property
    def coverage_target(self) -> float:
        """Target coverage probability (1 - alpha)."""
        return 1 - self.alpha

    @property
    def is_calibrated(self) -> bool:
        """True if calibrator has been calibrated."""
        return self._calibrated

    def calibrate(
        self,
        scores: List[Dict[str, float]],
        labels: List[str],
    ) -> CalibrationResult:
        """
        Calibrate for classification using softmax scores.

        Args:
            scores: List of {class: probability} dicts from model
            labels: True labels for calibration set

        Returns:
            CalibrationResult with threshold and stats
        """
        if len(scores) != len(labels):
            raise ValueError("Scores and labels must have same length")

        if not scores:
            raise ValueError("Cannot calibrate with empty data")

        # Compute conformity scores: 1 - softmax(true class)
        # Higher score = model was less confident about true class
        conformity_scores = []
        for score_dict, label in zip(scores, labels):
            true_prob = score_dict.get(label, 0.0)
            conformity_scores.append(1 - true_prob)

        # Compute threshold as quantile of conformity scores
        self._threshold = compute_quantile(conformity_scores, self.coverage_target)

        # Compute empirical coverage on calibration set
        covered = sum(1 for s in conformity_scores if s <= self._threshold)
        empirical_coverage = covered / len(conformity_scores)

        self._calibrated = True
        self._calibration_result = CalibrationResult(
            method=self.method,
            alpha=self.alpha,
            threshold=self._threshold,
            n_calibration=len(scores),
            empirical_coverage=empirical_coverage,
            metadata={
                "min_conformity": min(conformity_scores),
                "max_conformity": max(conformity_scores),
                "mean_conformity": sum(conformity_scores) / len(conformity_scores),
            }
        )

        return self._calibration_result

    def calibrate_regression(
        self,
        lower_quantiles: List[float],
        upper_quantiles: List[float],
        true_values: List[float],
    ) -> CalibrationResult:
        """
        Calibrate for regression using quantile predictions.

        Implements Conformalized Quantile Regression (CQR).

        Args:
            lower_quantiles: Model's lower quantile predictions (e.g., 5th percentile)
            upper_quantiles: Model's upper quantile predictions (e.g., 95th percentile)
            true_values: True values for calibration set

        Returns:
            CalibrationResult with threshold and stats
        """
        if not (len(lower_quantiles) == len(upper_quantiles) == len(true_values)):
            raise ValueError("All inputs must have same length")

        if not lower_quantiles:
            raise ValueError("Cannot calibrate with empty data")

        # Compute conformity scores for CQR:
        # max(lower - y, y - upper)
        # Score > 0 means true value outside predicted interval
        conformity_scores = []
        for lower, upper, y in zip(lower_quantiles, upper_quantiles, true_values):
            score = max(lower - y, y - upper)
            conformity_scores.append(score)

        # Threshold to add/subtract from intervals
        self._threshold = compute_quantile(conformity_scores, self.coverage_target)

        # Compute empirical coverage
        covered = sum(1 for s in conformity_scores if s <= self._threshold)
        empirical_coverage = covered / len(conformity_scores)

        self._calibrated = True
        self._calibration_result = CalibrationResult(
            method=CalibrationMethod.CQR,
            alpha=self.alpha,
            threshold=self._threshold,
            n_calibration=len(true_values),
            empirical_coverage=empirical_coverage,
            metadata={
                "min_conformity": min(conformity_scores),
                "max_conformity": max(conformity_scores),
                "mean_interval_width": sum(u - l for l, u in zip(lower_quantiles, upper_quantiles)) / len(lower_quantiles),
            }
        )

        return self._calibration_result

    def predict_set(
        self,
        scores: Dict[str, float],
        class_names: Optional[List[str]] = None,
    ) -> PredictionSet:
        """
        Generate prediction set for classification.

        Args:
            scores: {class: probability} dict from model
            class_names: Optional list of all possible classes

        Returns:
            PredictionSet containing classes that cannot be ruled out
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_set()")

        if class_names is None:
            class_names = list(scores.keys())

        # Include class if conformity score <= threshold
        # Conformity score = 1 - probability
        prediction_classes = []
        for cls in class_names:
            prob = scores.get(cls, 0.0)
            conformity = 1 - prob
            if conformity <= self._threshold:
                prediction_classes.append(cls)

        # Sort by probability (highest first)
        prediction_classes.sort(key=lambda c: scores.get(c, 0), reverse=True)

        return PredictionSet(
            classes=prediction_classes,
            scores=scores,
            threshold=self._threshold,
            alpha=self.alpha,
            set_size=len(prediction_classes),
        )

    def predict_interval(
        self,
        lower_quantile: float,
        upper_quantile: float,
        point_estimate: Optional[float] = None,
    ) -> PredictionInterval:
        """
        Generate prediction interval for regression.

        Args:
            lower_quantile: Model's lower quantile prediction
            upper_quantile: Model's upper quantile prediction
            point_estimate: Optional point prediction (defaults to midpoint)

        Returns:
            PredictionInterval with coverage guarantee
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate_regression() before predict_interval()")

        # Expand interval by calibrated threshold
        calibrated_lower = lower_quantile - self._threshold
        calibrated_upper = upper_quantile + self._threshold

        if point_estimate is None:
            point_estimate = (lower_quantile + upper_quantile) / 2

        return PredictionInterval(
            lower=calibrated_lower,
            upper=calibrated_upper,
            point_estimate=point_estimate,
            alpha=self.alpha,
        )


# =============================================================================
# SPECIALIZED CALIBRATORS
# =============================================================================

class AdaptivePredictionSets(ConformalCalibrator):
    """
    Adaptive Prediction Sets (APS) for classification.

    APS adds classes to the prediction set in order of decreasing
    probability until the cumulative probability exceeds the threshold.
    This creates smaller sets when the model is confident and larger
    sets when uncertain.

    Regularization Option (RAPS):
        Set regularization_weight > 0 to penalize large sets, producing
        smaller but still valid prediction sets.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        regularization_weight: float = 0.0,
    ):
        super().__init__(alpha=alpha, method=CalibrationMethod.APS)
        self.regularization_weight = regularization_weight
        if regularization_weight > 0:
            self.method = CalibrationMethod.RAPS

    def predict_set(
        self,
        scores: Dict[str, float],
        class_names: Optional[List[str]] = None,
    ) -> PredictionSet:
        """
        Generate adaptive prediction set.

        Classes are added in order of decreasing probability until
        the cumulative score exceeds the calibrated threshold.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_set()")

        if class_names is None:
            class_names = list(scores.keys())

        # Sort classes by probability (descending)
        sorted_classes = sorted(class_names, key=lambda c: scores.get(c, 0), reverse=True)

        # Add classes until cumulative >= threshold
        prediction_classes = []
        cumulative = 0.0

        for cls in sorted_classes:
            prob = scores.get(cls, 0.0)
            cumulative += prob

            # Regularization penalty for set size
            penalty = self.regularization_weight * len(prediction_classes)

            if cumulative - penalty >= self._threshold:
                prediction_classes.append(cls)
                break
            prediction_classes.append(cls)

        return PredictionSet(
            classes=prediction_classes,
            scores=scores,
            threshold=self._threshold,
            alpha=self.alpha,
            set_size=len(prediction_classes),
        )


class ConformizedQuantileRegression(ConformalCalibrator):
    """
    Conformalized Quantile Regression (CQR) for regression.

    CQR takes quantile predictions from any model and calibrates
    them to provide valid coverage guarantees. The intervals adapt
    to heteroscedasticity - wider where prediction is harder.

    Typical Usage:
        1. Train a quantile regression model to predict (e.g.) 5th and 95th percentiles
        2. Calibrate CQR on held-out data
        3. Apply calibration to expand/contract intervals for guaranteed coverage
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha=alpha, method=CalibrationMethod.CQR)

    def calibrate(
        self,
        lower_quantiles: List[float],
        upper_quantiles: List[float],
        true_values: List[float],
    ) -> CalibrationResult:
        """Calibrate CQR - delegates to calibrate_regression."""
        return self.calibrate_regression(lower_quantiles, upper_quantiles, true_values)

    def predict(
        self,
        lower_quantile: float,
        upper_quantile: float,
        point_estimate: Optional[float] = None,
    ) -> PredictionInterval:
        """Generate calibrated prediction interval."""
        return self.predict_interval(lower_quantile, upper_quantile, point_estimate)
