"""
TRUST Platform - Drift Monitor Module
======================================

Monitors for distribution drift to ensure conformal prediction guarantees remain valid.

Why Drift Monitoring?
    Conformal prediction guarantees coverage under the exchangeability assumption:
    calibration data and test data come from the same distribution. When the
    distribution shifts (drift), coverage guarantees may become invalid.

    In healthcare AI, drift is common:
    - Seasonal variations (flu season changes chest X-ray patterns)
    - Equipment changes (new MRI scanner has different noise characteristics)
    - Population shifts (hospital serves different demographics over time)
    - Practice changes (new clinical protocols affect documentation)

Types of Drift:

1. Covariate Drift (Input Drift)
   - Distribution of inputs P(X) changes
   - Model may still be correct for any given input
   - Example: More elderly patients → different image characteristics

2. Label Drift (Prior Drift)
   - Distribution of labels P(Y) changes
   - Example: COVID pandemic → more pneumonia cases

3. Concept Drift
   - Relationship P(Y|X) changes
   - Same inputs should now map to different outputs
   - Example: Updated diagnostic criteria change what "positive" means

Detection Methods:

1. Statistical Tests
   - Kolmogorov-Smirnov test for continuous features
   - Chi-squared test for categorical features
   - Population Stability Index (PSI)

2. Model-Based Detection
   - Train classifier to distinguish reference vs current data
   - High accuracy = significant drift

3. Coverage Monitoring
   - Track empirical coverage over time
   - Significant drop below target indicates drift

Example:
    >>> monitor = DriftMonitor(reference_data=calibration_set)
    >>> for batch in production_stream:
    ...     drift_result = monitor.check(batch)
    ...     if drift_result.is_significant:
    ...         print(f"DRIFT DETECTED: {drift_result.drift_type}")
    ...         # Trigger recalibration
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
from collections import deque


__all__ = [
    # Enums
    'DriftType',
    'DriftSeverity',
    # Data classes
    'DriftResult',
    'DriftAlert',
    'MonitoringWindow',
    # Main classes
    'DriftMonitor',
    'CoverageMonitor',
    'FeatureDriftDetector',
    # Utilities
    'compute_psi',
    'ks_statistic',
]


# =============================================================================
# ENUMS
# =============================================================================

class DriftType(str, Enum):
    """Type of distribution drift detected."""
    NONE = "none"
    COVARIATE = "covariate"      # Input distribution changed
    LABEL = "label"              # Output distribution changed
    CONCEPT = "concept"          # Input-output relationship changed
    COVERAGE = "coverage"        # Conformal coverage degraded
    UNKNOWN = "unknown"          # Drift detected but type unclear


class DriftSeverity(str, Enum):
    """Severity level of detected drift."""
    NONE = "none"
    LOW = "low"           # Minor drift, monitor closely
    MEDIUM = "medium"     # Noticeable drift, consider recalibration
    HIGH = "high"         # Significant drift, recalibration recommended
    CRITICAL = "critical" # Severe drift, immediate action required


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DriftResult:
    """Result of drift detection check."""
    drift_type: DriftType
    severity: DriftSeverity
    is_significant: bool
    statistic: float  # Test statistic value
    p_value: Optional[float]  # p-value if applicable
    threshold: float  # Threshold used for significance
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_recalibration(self) -> bool:
        """True if drift is severe enough to warrant recalibration."""
        return self.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""
    drift_result: DriftResult
    timestamp: str  # ISO format timestamp
    window_size: int  # Number of samples in detection window
    recommendation: str  # Recommended action
    affected_features: List[str] = field(default_factory=list)


@dataclass
class MonitoringWindow:
    """Statistics for a monitoring window."""
    start_idx: int
    end_idx: int
    n_samples: int
    coverage: float
    mean_set_size: float  # For classification
    mean_interval_width: float  # For regression
    drift_detected: bool


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_psi(
    reference: List[float],
    current: List[float],
    n_bins: int = 10
) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI measures the shift in distribution between reference and current data.

    Interpretation:
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.2: Moderate shift
        PSI >= 0.2: Significant shift

    Args:
        reference: Reference distribution values
        current: Current distribution values
        n_bins: Number of bins for histogram

    Returns:
        PSI value
    """
    if not reference or not current:
        return 0.0

    # Compute bin edges from reference
    min_val = min(min(reference), min(current))
    max_val = max(max(reference), max(current))

    if max_val == min_val:
        return 0.0

    bin_width = (max_val - min_val) / n_bins
    bins = [min_val + i * bin_width for i in range(n_bins + 1)]
    bins[-1] = max_val + 1e-10  # Ensure max value is included

    def histogram(values: List[float]) -> List[float]:
        counts = [0] * n_bins
        for v in values:
            for i in range(n_bins):
                if bins[i] <= v < bins[i + 1]:
                    counts[i] += 1
                    break
        # Convert to proportions with smoothing
        total = sum(counts)
        return [(c + 0.0001) / (total + 0.0001 * n_bins) for c in counts]

    ref_hist = histogram(reference)
    cur_hist = histogram(current)

    # PSI = sum((cur - ref) * ln(cur/ref))
    psi = 0.0
    for r, c in zip(ref_hist, cur_hist):
        if r > 0 and c > 0:
            psi += (c - r) * math.log(c / r)

    return psi


def ks_statistic(
    reference: List[float],
    current: List[float]
) -> Tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov statistic for two-sample test.

    Args:
        reference: Reference distribution values
        current: Current distribution values

    Returns:
        Tuple of (KS statistic, approximate p-value)
    """
    if not reference or not current:
        return 0.0, 1.0

    # Sort both samples
    ref_sorted = sorted(reference)
    cur_sorted = sorted(current)

    # Combine and sort all values
    all_values = sorted(set(ref_sorted + cur_sorted))

    # Compute ECDFs
    def ecdf(sorted_data: List[float], x: float) -> float:
        count = sum(1 for v in sorted_data if v <= x)
        return count / len(sorted_data)

    # Find maximum difference
    max_diff = 0.0
    for x in all_values:
        diff = abs(ecdf(ref_sorted, x) - ecdf(cur_sorted, x))
        max_diff = max(max_diff, diff)

    # Approximate p-value using asymptotic distribution
    n1, n2 = len(reference), len(current)
    en = math.sqrt(n1 * n2 / (n1 + n2))

    # Kolmogorov distribution approximation
    lambda_ks = (en + 0.12 + 0.11 / en) * max_diff

    if lambda_ks <= 0:
        p_value = 1.0
    else:
        # Asymptotic p-value approximation
        p_value = 2 * math.exp(-2 * lambda_ks * lambda_ks)
        p_value = min(1.0, max(0.0, p_value))

    return max_diff, p_value


# =============================================================================
# DRIFT MONITOR - MAIN CLASS
# =============================================================================

class DriftMonitor:
    """
    Main drift monitoring class for detecting distribution shifts.

    The monitor maintains reference statistics and compares incoming
    data batches to detect significant drift. When drift is detected,
    it can trigger recalibration of conformal predictors.

    Architecture Decision:
        The drift monitor is domain-agnostic. It works with numerical
        features and coverage statistics. Domain-specific interpretation
        (what features mean, what actions to take) is handled by the
        governance pipelines.

    Example:
        >>> # Initialize with reference data
        >>> monitor = DriftMonitor()
        >>> monitor.set_reference(calibration_features)
        >>>
        >>> # Check for drift in production
        >>> for batch in production_batches:
        ...     result = monitor.check(batch)
        ...     if result.needs_recalibration:
        ...         recalibrate_model()
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1,
        coverage_threshold: float = 0.05,
        window_size: int = 100,
    ):
        """
        Initialize drift monitor.

        Args:
            psi_threshold: PSI threshold for significant drift (default 0.2)
            ks_threshold: KS statistic threshold (default 0.1)
            coverage_threshold: Coverage drop threshold (default 0.05 = 5%)
            window_size: Window size for rolling statistics
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.coverage_threshold = coverage_threshold
        self.window_size = window_size

        self._reference_data: Dict[str, List[float]] = {}
        self._reference_stats: Dict[str, Dict[str, float]] = {}
        self._current_window: deque = deque(maxlen=window_size)
        self._alerts: List[DriftAlert] = []

    def set_reference(
        self,
        features: Dict[str, List[float]],
    ) -> None:
        """
        Set reference distribution from calibration data.

        Args:
            features: Dict mapping feature names to values
        """
        self._reference_data = features
        self._reference_stats = {}

        for name, values in features.items():
            if values:
                self._reference_stats[name] = {
                    "mean": sum(values) / len(values),
                    "std": self._compute_std(values),
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }

    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def check(
        self,
        features: Dict[str, List[float]],
        feature_names: Optional[List[str]] = None,
    ) -> DriftResult:
        """
        Check for drift in current batch.

        Args:
            features: Dict mapping feature names to current values
            feature_names: Optional subset of features to check

        Returns:
            DriftResult with detection outcome
        """
        if not self._reference_data:
            return DriftResult(
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                is_significant=False,
                statistic=0.0,
                p_value=1.0,
                threshold=self.psi_threshold,
                details={"error": "No reference data set"},
            )

        if feature_names is None:
            feature_names = list(self._reference_data.keys())

        # Check each feature
        drift_scores = {}
        drifted_features = []

        for name in feature_names:
            if name not in self._reference_data or name not in features:
                continue

            ref_values = self._reference_data[name]
            cur_values = features[name]

            if not ref_values or not cur_values:
                continue

            # Compute PSI
            psi = compute_psi(ref_values, cur_values)
            drift_scores[name] = {"psi": psi}

            # Compute KS statistic
            ks_stat, ks_pval = ks_statistic(ref_values, cur_values)
            drift_scores[name]["ks_statistic"] = ks_stat
            drift_scores[name]["ks_pvalue"] = ks_pval

            # Check if significant
            if psi >= self.psi_threshold or ks_stat >= self.ks_threshold:
                drifted_features.append(name)

        # Determine overall drift
        if not drifted_features:
            return DriftResult(
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                is_significant=False,
                statistic=0.0,
                p_value=1.0,
                threshold=self.psi_threshold,
                details={"feature_scores": drift_scores},
            )

        # Compute aggregate statistic
        max_psi = max(drift_scores[f]["psi"] for f in drifted_features)

        # Determine severity
        if max_psi >= 0.5:
            severity = DriftSeverity.CRITICAL
        elif max_psi >= 0.3:
            severity = DriftSeverity.HIGH
        elif max_psi >= 0.2:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        return DriftResult(
            drift_type=DriftType.COVARIATE,
            severity=severity,
            is_significant=True,
            statistic=max_psi,
            p_value=None,
            threshold=self.psi_threshold,
            details={
                "feature_scores": drift_scores,
                "drifted_features": drifted_features,
                "n_drifted": len(drifted_features),
                "n_checked": len(feature_names),
            },
        )

    def create_alert(
        self,
        drift_result: DriftResult,
        timestamp: str = "now",
    ) -> DriftAlert:
        """Create and store an alert for detected drift."""
        if drift_result.severity == DriftSeverity.CRITICAL:
            recommendation = "IMMEDIATE: Stop accepting predictions, recalibrate immediately"
        elif drift_result.severity == DriftSeverity.HIGH:
            recommendation = "Recalibrate model within 24 hours"
        elif drift_result.severity == DriftSeverity.MEDIUM:
            recommendation = "Schedule recalibration within 1 week"
        else:
            recommendation = "Continue monitoring, no action needed"

        alert = DriftAlert(
            drift_result=drift_result,
            timestamp=timestamp,
            window_size=self.window_size,
            recommendation=recommendation,
            affected_features=drift_result.details.get("drifted_features", []),
        )

        self._alerts.append(alert)
        return alert

    @property
    def alerts(self) -> List[DriftAlert]:
        """Get all stored alerts."""
        return self._alerts.copy()

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self._alerts.clear()


# =============================================================================
# COVERAGE MONITOR
# =============================================================================

class CoverageMonitor:
    """
    Monitors empirical coverage to detect when conformal guarantees may be violated.

    This is the most direct way to detect concept drift - if coverage drops
    below the target, something has changed about the data distribution.

    Example:
        >>> monitor = CoverageMonitor(target_coverage=0.9, window_size=100)
        >>> for prediction, true_label in production_stream:
        ...     monitor.update(true_label in prediction.classes)
        ...     if monitor.is_coverage_degraded():
        ...         trigger_recalibration()
    """

    def __init__(
        self,
        target_coverage: float = 0.9,
        window_size: int = 100,
        tolerance: float = 0.05,
    ):
        """
        Initialize coverage monitor.

        Args:
            target_coverage: Expected coverage (1 - alpha)
            window_size: Rolling window size for coverage calculation
            tolerance: Acceptable coverage drop before flagging (default 5%)
        """
        self.target_coverage = target_coverage
        self.window_size = window_size
        self.tolerance = tolerance

        self._covered: deque = deque(maxlen=window_size)
        self._windows: List[MonitoringWindow] = []
        self._total_samples = 0
        self._total_covered = 0

    def update(self, is_covered: bool) -> None:
        """
        Update with a new prediction result.

        Args:
            is_covered: True if true value was in prediction set/interval
        """
        self._covered.append(1 if is_covered else 0)
        self._total_samples += 1
        if is_covered:
            self._total_covered += 1

    @property
    def current_coverage(self) -> float:
        """Current coverage in the rolling window."""
        if not self._covered:
            return 1.0
        return sum(self._covered) / len(self._covered)

    @property
    def overall_coverage(self) -> float:
        """Overall coverage across all samples."""
        if self._total_samples == 0:
            return 1.0
        return self._total_covered / self._total_samples

    @property
    def coverage_gap(self) -> float:
        """Gap between target and current coverage."""
        return self.target_coverage - self.current_coverage

    def is_coverage_degraded(self) -> bool:
        """
        Check if coverage has dropped significantly below target.

        Returns:
            True if current coverage is below (target - tolerance)
        """
        if len(self._covered) < self.window_size // 2:
            # Not enough data for reliable estimate
            return False
        return self.current_coverage < (self.target_coverage - self.tolerance)

    def check(self) -> DriftResult:
        """
        Check for coverage drift.

        Returns:
            DriftResult indicating coverage status
        """
        if len(self._covered) < self.window_size // 2:
            return DriftResult(
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                is_significant=False,
                statistic=0.0,
                p_value=None,
                threshold=self.tolerance,
                details={"n_samples": len(self._covered), "min_required": self.window_size // 2},
            )

        gap = self.coverage_gap

        if gap <= 0:
            severity = DriftSeverity.NONE
            is_significant = False
        elif gap < self.tolerance:
            severity = DriftSeverity.LOW
            is_significant = False
        elif gap < self.tolerance * 2:
            severity = DriftSeverity.MEDIUM
            is_significant = True
        elif gap < self.tolerance * 3:
            severity = DriftSeverity.HIGH
            is_significant = True
        else:
            severity = DriftSeverity.CRITICAL
            is_significant = True

        return DriftResult(
            drift_type=DriftType.COVERAGE if is_significant else DriftType.NONE,
            severity=severity,
            is_significant=is_significant,
            statistic=gap,
            p_value=None,
            threshold=self.tolerance,
            details={
                "target_coverage": self.target_coverage,
                "current_coverage": self.current_coverage,
                "overall_coverage": self.overall_coverage,
                "window_size": len(self._covered),
                "total_samples": self._total_samples,
            },
        )

    def save_window(self, mean_set_size: float = 0.0, mean_interval_width: float = 0.0) -> MonitoringWindow:
        """Save current window statistics."""
        window = MonitoringWindow(
            start_idx=self._total_samples - len(self._covered),
            end_idx=self._total_samples,
            n_samples=len(self._covered),
            coverage=self.current_coverage,
            mean_set_size=mean_set_size,
            mean_interval_width=mean_interval_width,
            drift_detected=self.is_coverage_degraded(),
        )
        self._windows.append(window)
        return window

    @property
    def windows(self) -> List[MonitoringWindow]:
        """Get all saved monitoring windows."""
        return self._windows.copy()


# =============================================================================
# FEATURE DRIFT DETECTOR
# =============================================================================

class FeatureDriftDetector:
    """
    Detects drift in individual features.

    Useful for identifying which specific inputs are changing,
    which can help diagnose the cause of drift.
    """

    def __init__(self, psi_threshold: float = 0.2, ks_threshold: float = 0.1):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self._reference: Dict[str, List[float]] = {}

    def set_reference(self, feature_name: str, values: List[float]) -> None:
        """Set reference distribution for a feature."""
        self._reference[feature_name] = values

    def check_feature(
        self,
        feature_name: str,
        current_values: List[float]
    ) -> DriftResult:
        """Check drift for a single feature."""
        if feature_name not in self._reference:
            return DriftResult(
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                is_significant=False,
                statistic=0.0,
                p_value=None,
                threshold=self.psi_threshold,
                details={"error": f"No reference for feature {feature_name}"},
            )

        ref_values = self._reference[feature_name]
        psi = compute_psi(ref_values, current_values)
        ks_stat, ks_pval = ks_statistic(ref_values, current_values)

        is_significant = psi >= self.psi_threshold or ks_stat >= self.ks_threshold

        if psi >= 0.5:
            severity = DriftSeverity.CRITICAL
        elif psi >= 0.3:
            severity = DriftSeverity.HIGH
        elif psi >= 0.2:
            severity = DriftSeverity.MEDIUM
        elif psi >= 0.1:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        return DriftResult(
            drift_type=DriftType.COVARIATE if is_significant else DriftType.NONE,
            severity=severity,
            is_significant=is_significant,
            statistic=psi,
            p_value=ks_pval,
            threshold=self.psi_threshold,
            details={
                "feature": feature_name,
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
            },
        )
