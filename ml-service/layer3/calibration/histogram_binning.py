"""
Histogram binning calibrator for Layer 2 risk scores.

An alternative to conformal calibration that partitions the score range
into equal-width bins, computes the empirical positive rate in each bin,
and maps new scores to the bin's observed rate.

This is a post-hoc recalibration method (Zadrozny & Elkan 2001) that
improves Expected Calibration Error (ECE) but does *not* provide the
finite-sample coverage guarantee of conformal prediction.

Pure numpy — no GPU required.
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BinStats:
    """Statistics for a single histogram bin."""
    bin_lower: float
    bin_upper: float
    count: int
    positive_rate: float  # fraction of positives in this bin


@dataclass
class _HistogramState:
    """Serialisable internal state."""
    n_bins: int
    min_bin_count: int
    bins: List[dict]  # list of BinStats as dicts
    n_calibration: int


class HistogramCalibrator:
    """
    Histogram binning calibrator.

    Divides [0, 1] into *n_bins* equal-width buckets. For each bucket the
    empirical positive rate on the calibration set is computed.  New
    scores are mapped to the positive rate of their bin.

    If a bin has fewer than *min_bin_count* samples it falls back to the
    raw score (insufficient data to recalibrate).
    """

    def __init__(self, n_bins: int = 10, min_bin_count: int = 5):
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self._n_bins = n_bins
        self._min_bin_count = min_bin_count
        self._bins: Optional[List[BinStats]] = None
        self._n_calibration: int = 0

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._bins is not None

    @property
    def n_bins(self) -> int:
        return self._n_bins

    @property
    def bins(self) -> List[BinStats]:
        if self._bins is None:
            raise RuntimeError("Calibrator has not been fitted yet")
        return list(self._bins)

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        predictions: List[float],
        labels: List[int],
    ) -> "HistogramCalibrator":
        """
        Fit histogram bins on a calibration set.

        Parameters
        ----------
        predictions : list[float]
            Raw risk scores (0.0 – 1.0).
        labels : list[int]
            Binary labels (1 = review needed, 0 = not needed).
        """
        if len(predictions) != len(labels):
            raise ValueError("predictions and labels must have the same length")
        if len(predictions) == 0:
            raise ValueError("Cannot fit with empty data")

        width = 1.0 / self._n_bins
        bins: List[BinStats] = []

        for i in range(self._n_bins):
            lo = i * width
            hi = (i + 1) * width
            # Include right edge in the last bin
            in_bin = [
                (p, l)
                for p, l in zip(predictions, labels)
                if (lo <= p < hi) or (i == self._n_bins - 1 and p == hi)
            ]
            count = len(in_bin)
            if count > 0:
                pos_rate = sum(l for _, l in in_bin) / count
            else:
                pos_rate = (lo + hi) / 2  # fallback: bin midpoint

            bins.append(BinStats(
                bin_lower=round(lo, 10),
                bin_upper=round(hi, 10),
                count=count,
                positive_rate=round(pos_rate, 10),
            ))

        self._bins = bins
        self._n_calibration = len(predictions)
        logger.info(
            "Fitted histogram calibrator: %d bins, %d samples",
            self._n_bins, self._n_calibration,
        )
        return self

    # ── Calibrate ────────────────────────────────────────────────────────

    def calibrate(self, score: float) -> float:
        """
        Map a raw score to its calibrated value.

        Returns the empirical positive rate of the bin the score falls
        into.  Falls back to the raw score if the bin has too few samples.
        """
        if self._bins is None:
            raise RuntimeError("Calibrator has not been fitted yet")

        b = self._find_bin(score)
        if b.count < self._min_bin_count:
            return score  # insufficient data
        return b.positive_rate

    def calibrate_batch(self, scores: List[float]) -> List[float]:
        """Calibrate a list of scores."""
        return [self.calibrate(s) for s in scores]

    # ── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self._bins is None:
            raise RuntimeError("Cannot save unfitted calibrator")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        state = _HistogramState(
            n_bins=self._n_bins,
            min_bin_count=self._min_bin_count,
            bins=[asdict(b) for b in self._bins],
            n_calibration=self._n_calibration,
        )
        with open(out, "w") as f:
            json.dump(asdict(state), f, indent=2)
        logger.info("Saved histogram calibrator to %s", path)

    def load(self, path: str) -> "HistogramCalibrator":
        with open(path) as f:
            raw = json.load(f)
        self._n_bins = raw["n_bins"]
        self._min_bin_count = raw["min_bin_count"]
        self._n_calibration = raw["n_calibration"]
        self._bins = [
            BinStats(**b) for b in raw["bins"]
        ]
        logger.info(
            "Loaded histogram calibrator from %s (%d bins, n=%d)",
            path, self._n_bins, self._n_calibration,
        )
        return self

    # ── Internals ────────────────────────────────────────────────────────

    def _find_bin(self, score: float) -> BinStats:
        """Return the bin a score falls into."""
        score = max(0.0, min(1.0, score))
        idx = min(int(score * self._n_bins), self._n_bins - 1)
        return self._bins[idx]
