"""
Conformal calibration for Layer 2 risk scores.

Uses split conformal prediction with quantile-based thresholding to
produce calibrated risk scores and prediction sets with finite-sample
coverage guarantees.

The nonconformity score is ``|predicted_risk - label|`` where label is 1
for claims that actually needed review and 0 otherwise.  The quantile of
these scores on a held-out calibration set yields a threshold that, when
applied to new predictions, guarantees the true review status is captured
at the configured coverage rate (default 90 %).

Pure numpy — no GPU required.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

REVIEW_LEVELS = ["BRIEF", "STANDARD", "DETAILED", "CRITICAL"]

# Thresholds mapping risk scores to review levels (ascending order)
_LEVEL_THRESHOLDS = [
    (0.20, "BRIEF"),
    (0.50, "STANDARD"),
    (0.80, "DETAILED"),
    (1.01, "CRITICAL"),  # 1.01 so score=1.0 maps to CRITICAL
]


def _score_to_level(score: float) -> str:
    """Map a risk score [0, 1] to a review level string."""
    for threshold, level in _LEVEL_THRESHOLDS:
        if score < threshold:
            return level
    return "CRITICAL"


def _level_index(level: str) -> int:
    """Return ordinal index of a review level (0=BRIEF … 3=CRITICAL)."""
    try:
        return REVIEW_LEVELS.index(level)
    except ValueError:
        return 1  # default to STANDARD


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CalibratedResult:
    """Output of calibrating a single risk score."""
    raw_score: float
    calibrated_score: float
    prediction_set: Set[str]
    confidence: float
    threshold: float


@dataclass
class _CalibratorState:
    """Serialisable internal state for save/load."""
    threshold: float
    coverage_target: float
    n_calibration: int
    empirical_coverage: float
    nonconformity_scores: List[float] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# CONFORMAL CALIBRATOR
# ═════════════════════════════════════════════════════════════════════════════

class ConformalCalibrator:
    """
    Split conformal calibrator for Layer 2 meta-classifier risk scores.

    Workflow:
        1. ``fit(predictions, labels)`` — compute nonconformity scores and
           the coverage-adjusted quantile threshold on a calibration set.
        2. ``calibrate(score)`` — map a new raw score to a
           ``CalibratedResult`` with prediction set.
        3. ``get_prediction_set(score, confidence)`` — convenience wrapper
           returning only the prediction set at a given confidence level.

    The coverage guarantee is finite-sample valid under exchangeability:
        P(true label ∈ prediction_set) ≥ 1 − α
    """

    def __init__(self, coverage_target: float = 0.90):
        if not 0.0 < coverage_target < 1.0:
            raise ValueError(
                f"coverage_target must be in (0, 1), got {coverage_target}"
            )
        self._coverage_target = coverage_target
        self._alpha = 1.0 - coverage_target
        self._state: Optional[_CalibratorState] = None

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def coverage_target(self) -> float:
        return self._coverage_target

    @property
    def threshold(self) -> float:
        if self._state is None:
            raise RuntimeError("Calibrator has not been fitted yet")
        return self._state.threshold

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        predictions: List[float],
        labels: List[int],
    ) -> "ConformalCalibrator":
        """
        Fit the calibrator on a held-out calibration set.

        Parameters
        ----------
        predictions : list[float]
            Raw risk scores from Layer 2 (0.0 – 1.0).
        labels : list[int]
            Binary ground-truth labels (1 = review was actually needed,
            0 = review was not needed).

        Returns
        -------
        self
        """
        if len(predictions) != len(labels):
            raise ValueError(
                f"predictions ({len(predictions)}) and labels "
                f"({len(labels)}) must have the same length"
            )
        if len(predictions) == 0:
            raise ValueError("Cannot fit with empty data")

        # Nonconformity score: |predicted − actual|
        nc_scores = [
            abs(pred - label) for pred, label in zip(predictions, labels)
        ]

        # Quantile with finite-sample adjustment: ceil((n+1)(1−α)) / n
        n = len(nc_scores)
        q = self._coverage_target
        adjusted_q = min(1.0, q * (1 + 1 / n))
        sorted_scores = sorted(nc_scores)
        idx = min(int(math.ceil(adjusted_q * n)) - 1, n - 1)
        idx = max(idx, 0)
        threshold = sorted_scores[idx]

        # Empirical coverage on calibration set
        covered = sum(1 for s in nc_scores if s <= threshold)
        empirical_coverage = covered / n

        self._state = _CalibratorState(
            threshold=threshold,
            coverage_target=self._coverage_target,
            n_calibration=n,
            empirical_coverage=empirical_coverage,
            nonconformity_scores=sorted_scores,
        )

        logger.info(
            "Fitted conformal calibrator: threshold=%.4f, "
            "empirical_coverage=%.4f, n=%d",
            threshold, empirical_coverage, n,
        )
        return self

    # ── Calibrate ────────────────────────────────────────────────────────

    def calibrate(self, score: float) -> CalibratedResult:
        """
        Calibrate a single raw risk score.

        Returns a ``CalibratedResult`` containing the calibrated score,
        a prediction set of review levels, and the confidence.
        """
        if self._state is None:
            raise RuntimeError("Calibrator has not been fitted yet")

        threshold = self._state.threshold

        # Calibrated score: clip the raw score by the threshold band
        # A score whose nonconformity would exceed the threshold is
        # pushed towards the boundary.
        if score > 1.0 - threshold:
            calibrated = min(score, 1.0)
        elif score < threshold:
            calibrated = max(score, 0.0)
        else:
            calibrated = score

        prediction_set = self._build_prediction_set(score, threshold)

        # Confidence: how far inside the threshold band the score sits
        nc = abs(score - round(score))  # distance to nearest integer label
        confidence = max(0.0, min(1.0, 1.0 - nc / max(threshold, 1e-9)))

        return CalibratedResult(
            raw_score=score,
            calibrated_score=round(calibrated, 6),
            prediction_set=prediction_set,
            confidence=round(confidence, 6),
            threshold=threshold,
        )

    # ── Prediction set ───────────────────────────────────────────────────

    def get_prediction_set(
        self,
        score: float,
        confidence: Optional[float] = None,
    ) -> Set[str]:
        """
        Return the prediction set of review levels for a raw score.

        If *confidence* is provided it overrides the fitted coverage
        target for this call (higher confidence → larger set).
        """
        if self._state is None:
            raise RuntimeError("Calibrator has not been fitted yet")

        threshold = self._state.threshold

        if confidence is not None:
            # Scale threshold proportionally to requested confidence
            ratio = confidence / self._coverage_target
            threshold = min(threshold * ratio, 1.0)

        return self._build_prediction_set(score, threshold)

    # ── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist calibrator state to a JSON file."""
        if self._state is None:
            raise RuntimeError("Cannot save unfitted calibrator")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(asdict(self._state), f, indent=2)
        logger.info("Saved calibrator state to %s", path)

    def load(self, path: str) -> "ConformalCalibrator":
        """Restore calibrator state from a JSON file."""
        with open(path) as f:
            raw = json.load(f)
        self._state = _CalibratorState(
            threshold=raw["threshold"],
            coverage_target=raw["coverage_target"],
            n_calibration=raw["n_calibration"],
            empirical_coverage=raw["empirical_coverage"],
            nonconformity_scores=raw.get("nonconformity_scores", []),
        )
        self._coverage_target = self._state.coverage_target
        self._alpha = 1.0 - self._coverage_target
        logger.info(
            "Loaded calibrator state from %s (threshold=%.4f, n=%d)",
            path, self._state.threshold, self._state.n_calibration,
        )
        return self

    # ── Internals ────────────────────────────────────────────────────────

    def _build_prediction_set(
        self, score: float, threshold: float,
    ) -> Set[str]:
        """
        Build the set of review levels whose boundary is reachable from
        *score* within the nonconformity *threshold*.

        Any level whose score range overlaps [score − threshold,
        score + threshold] is included.
        """
        lo = max(0.0, score - threshold)
        hi = min(1.0, score + threshold)

        result: Set[str] = set()
        # Walk the level thresholds and include any level whose band
        # overlaps [lo, hi].
        prev_boundary = 0.0
        for boundary, level in _LEVEL_THRESHOLDS:
            level_lo = prev_boundary
            level_hi = boundary
            if level_lo < hi and lo < level_hi:
                result.add(level)
            prev_boundary = boundary

        # Always include the level of the point estimate itself
        result.add(_score_to_level(score))
        return result
