"""
Tests for Layer 3 conformal calibration.

Covers:
- Config loading and defaults
- ConformalCalibrator: fit, calibrate, prediction sets, coverage guarantee
- HistogramCalibrator: fit, calibrate, bin statistics
- Metrics: ECE, Brier score, coverage rate, calibration curve
- Save/load roundtrip for both calibrators
- Edge cases and error handling
"""

import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from layer3.config.calibration_config import (
    Layer3Config,
    ConformalConfig,
    HistogramConfig,
    load_config,
)
from layer3.calibration.conformal import (
    ConformalCalibrator,
    CalibratedResult,
    REVIEW_LEVELS,
    _score_to_level,
)
from layer3.calibration.histogram_binning import (
    HistogramCalibrator,
    BinStats,
)
from layer3.calibration.metrics import (
    expected_calibration_error,
    brier_score,
    coverage_rate,
    calibration_curve,
)


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

def _make_calibration_data(n=500, seed=42):
    """
    Generate synthetic calibration data.

    Scores near 0 are mostly label-0, scores near 1 are mostly label-1,
    with some noise so that calibration has work to do.
    """
    rng = random.Random(seed)
    predictions = []
    labels = []
    for _ in range(n):
        # Raw score with some miscalibration (shifted up by 0.15)
        true_prob = rng.random()
        label = 1 if rng.random() < true_prob else 0
        score = min(1.0, max(0.0, true_prob + 0.15 + rng.gauss(0, 0.10)))
        predictions.append(score)
        labels.append(label)
    return predictions, labels


def _make_well_calibrated_data(n=500, seed=99):
    """Data where predicted score ≈ true positive rate (low ECE)."""
    rng = random.Random(seed)
    preds, labs = [], []
    for _ in range(n):
        p = rng.random()
        lab = 1 if rng.random() < p else 0
        preds.append(p)
        labs.append(lab)
    return preds, labs


@pytest.fixture
def cal_data():
    return _make_calibration_data(500, seed=42)


@pytest.fixture
def held_out_data():
    """Separate test set from the same distribution."""
    return _make_calibration_data(300, seed=7)


@pytest.fixture
def fitted_calibrator(cal_data):
    preds, labels = cal_data
    cal = ConformalCalibrator(coverage_target=0.90)
    cal.fit(preds, labels)
    return cal


@pytest.fixture
def fitted_histogram(cal_data):
    preds, labels = cal_data
    hc = HistogramCalibrator(n_bins=10, min_bin_count=5)
    hc.fit(preds, labels)
    return hc


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_default_values(self):
        cfg = Layer3Config()
        assert cfg.conformal.coverage_target == 0.90
        assert cfg.conformal.min_calibration_samples == 100
        assert cfg.histogram.score_bins == 10

    def test_load_config_from_yaml(self):
        cfg = load_config()
        assert cfg.conformal.coverage_target == 0.90
        assert cfg.histogram.score_bins == 10
        assert cfg.persistence.filename == "calibrator_state.json"

    def test_load_config_missing_file(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg.conformal.coverage_target == 0.90  # defaults

    def test_review_levels_in_config(self):
        cfg = Layer3Config()
        assert cfg.conformal.review_levels == ["BRIEF", "STANDARD", "DETAILED", "CRITICAL"]


# ═════════════════════════════════════════════════════════════════════════════
# SCORE-TO-LEVEL MAPPING
# ═════════════════════════════════════════════════════════════════════════════

class TestScoreToLevel:
    def test_low_scores_map_to_brief(self):
        assert _score_to_level(0.0) == "BRIEF"
        assert _score_to_level(0.10) == "BRIEF"
        assert _score_to_level(0.19) == "BRIEF"

    def test_medium_scores_map_to_standard(self):
        assert _score_to_level(0.20) == "STANDARD"
        assert _score_to_level(0.35) == "STANDARD"
        assert _score_to_level(0.49) == "STANDARD"

    def test_high_scores_map_to_detailed(self):
        assert _score_to_level(0.50) == "DETAILED"
        assert _score_to_level(0.65) == "DETAILED"
        assert _score_to_level(0.79) == "DETAILED"

    def test_critical_scores(self):
        assert _score_to_level(0.80) == "CRITICAL"
        assert _score_to_level(0.95) == "CRITICAL"
        assert _score_to_level(1.0) == "CRITICAL"


# ═════════════════════════════════════════════════════════════════════════════
# CONFORMAL CALIBRATOR — FIT
# ═════════════════════════════════════════════════════════════════════════════

class TestConformalFit:
    def test_fit_sets_state(self, fitted_calibrator):
        assert fitted_calibrator.is_fitted
        assert fitted_calibrator.threshold > 0

    def test_coverage_target_preserved(self, fitted_calibrator):
        assert fitted_calibrator.coverage_target == 0.90

    def test_fit_returns_self(self, cal_data):
        preds, labels = cal_data
        cal = ConformalCalibrator(coverage_target=0.90)
        result = cal.fit(preds, labels)
        assert result is cal

    def test_fit_rejects_empty_data(self):
        cal = ConformalCalibrator()
        with pytest.raises(ValueError, match="empty"):
            cal.fit([], [])

    def test_fit_rejects_mismatched_lengths(self):
        cal = ConformalCalibrator()
        with pytest.raises(ValueError, match="same length"):
            cal.fit([0.5, 0.6], [1])

    def test_invalid_coverage_target(self):
        with pytest.raises(ValueError):
            ConformalCalibrator(coverage_target=0.0)
        with pytest.raises(ValueError):
            ConformalCalibrator(coverage_target=1.0)
        with pytest.raises(ValueError):
            ConformalCalibrator(coverage_target=-0.1)

    def test_threshold_is_nonnegative(self, fitted_calibrator):
        assert fitted_calibrator.threshold >= 0.0

    def test_threshold_before_fit_raises(self):
        cal = ConformalCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = cal.threshold


# ═════════════════════════════════════════════════════════════════════════════
# CONFORMAL CALIBRATOR — CALIBRATE
# ═════════════════════════════════════════════════════════════════════════════

class TestConformalCalibrate:
    def test_returns_calibrated_result(self, fitted_calibrator):
        result = fitted_calibrator.calibrate(0.5)
        assert isinstance(result, CalibratedResult)

    def test_result_has_all_fields(self, fitted_calibrator):
        result = fitted_calibrator.calibrate(0.75)
        assert result.raw_score == 0.75
        assert 0.0 <= result.calibrated_score <= 1.0
        assert isinstance(result.prediction_set, set)
        assert len(result.prediction_set) >= 1
        assert 0.0 <= result.confidence <= 1.0
        assert result.threshold > 0

    def test_prediction_set_contains_valid_levels(self, fitted_calibrator):
        for score in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            result = fitted_calibrator.calibrate(score)
            assert result.prediction_set.issubset(set(REVIEW_LEVELS))

    def test_prediction_set_includes_point_estimate_level(self, fitted_calibrator):
        for score in [0.05, 0.30, 0.60, 0.90]:
            result = fitted_calibrator.calibrate(score)
            expected_level = _score_to_level(score)
            assert expected_level in result.prediction_set

    def test_low_scores_have_small_sets(self, fitted_calibrator):
        result = fitted_calibrator.calibrate(0.05)
        # Very low score should not include CRITICAL
        assert "CRITICAL" not in result.prediction_set or fitted_calibrator.threshold > 0.75

    def test_calibrate_before_fit_raises(self):
        cal = ConformalCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.calibrate(0.5)


# ═════════════════════════════════════════════════════════════════════════════
# CONFORMAL CALIBRATOR — COVERAGE GUARANTEE
# ═════════════════════════════════════════════════════════════════════════════

class TestCoverageGuarantee:
    def test_coverage_on_calibration_set(self, cal_data, fitted_calibrator):
        """Coverage on the calibration set should meet or exceed target."""
        preds, labels = cal_data
        covered = 0
        for score, label in zip(preds, labels):
            result = fitted_calibrator.calibrate(score)
            # The true level for label 1 is determined by the score
            # For coverage, we check nonconformity: |score - label| <= threshold
            nc = abs(score - label)
            if nc <= fitted_calibrator.threshold:
                covered += 1
        actual_coverage = covered / len(preds)
        assert actual_coverage >= 0.89, f"Coverage {actual_coverage} below 0.89"

    def test_coverage_on_held_out_data(self, fitted_calibrator, held_out_data):
        """Coverage on held-out data should approximate the target."""
        preds, labels = held_out_data
        covered = 0
        for score, label in zip(preds, labels):
            nc = abs(score - label)
            if nc <= fitted_calibrator.threshold:
                covered += 1
        actual_coverage = covered / len(preds)
        # Allow some slack for finite sample
        assert actual_coverage >= 0.85, f"Held-out coverage {actual_coverage} below 0.85"

    def test_prediction_sets_contain_true_label(self, fitted_calibrator, held_out_data):
        """
        Prediction sets should contain the true review level at roughly
        the target rate.
        """
        preds, labels = held_out_data
        covered = 0
        for score, label in zip(preds, labels):
            result = fitted_calibrator.calibrate(score)
            # Determine what the "true" review level would be for this label
            if label == 1:
                true_level = _score_to_level(score)
            else:
                true_level = "BRIEF"
            if true_level in result.prediction_set:
                covered += 1
        rate = covered / len(preds)
        # Should be well above random
        assert rate >= 0.70, f"Prediction set coverage {rate} below 0.70"

    def test_higher_coverage_target_gives_larger_sets(self, cal_data):
        """90 % target should produce smaller sets on average than 99 %."""
        preds, labels = cal_data
        cal_90 = ConformalCalibrator(coverage_target=0.90)
        cal_90.fit(preds, labels)
        cal_99 = ConformalCalibrator(coverage_target=0.99)
        cal_99.fit(preds, labels)

        sizes_90 = [len(cal_90.calibrate(s).prediction_set) for s in preds[:100]]
        sizes_99 = [len(cal_99.calibrate(s).prediction_set) for s in preds[:100]]

        avg_90 = sum(sizes_90) / len(sizes_90)
        avg_99 = sum(sizes_99) / len(sizes_99)
        assert avg_99 >= avg_90, (
            f"99% sets (avg={avg_99:.2f}) should be >= 90% sets (avg={avg_90:.2f})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# CONFORMAL CALIBRATOR — GET PREDICTION SET
# ═════════════════════════════════════════════════════════════════════════════

class TestGetPredictionSet:
    def test_returns_set(self, fitted_calibrator):
        ps = fitted_calibrator.get_prediction_set(0.5)
        assert isinstance(ps, set)
        assert len(ps) >= 1

    def test_higher_confidence_grows_set(self, fitted_calibrator):
        ps_low = fitted_calibrator.get_prediction_set(0.5, confidence=0.80)
        ps_high = fitted_calibrator.get_prediction_set(0.5, confidence=0.99)
        assert len(ps_high) >= len(ps_low)

    def test_unfitted_raises(self):
        cal = ConformalCalibrator()
        with pytest.raises(RuntimeError):
            cal.get_prediction_set(0.5)


# ═════════════════════════════════════════════════════════════════════════════
# CONFORMAL CALIBRATOR — SAVE / LOAD
# ═════════════════════════════════════════════════════════════════════════════

class TestConformalSaveLoad:
    def test_roundtrip(self, fitted_calibrator):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            fitted_calibrator.save(path)

            loaded = ConformalCalibrator()
            loaded.load(path)

            assert loaded.is_fitted
            assert loaded.threshold == pytest.approx(fitted_calibrator.threshold)
            assert loaded.coverage_target == fitted_calibrator.coverage_target

    def test_loaded_calibrator_produces_same_results(self, fitted_calibrator):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            fitted_calibrator.save(path)

            loaded = ConformalCalibrator()
            loaded.load(path)

            for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
                r1 = fitted_calibrator.calibrate(score)
                r2 = loaded.calibrate(score)
                assert r1.calibrated_score == pytest.approx(r2.calibrated_score)
                assert r1.prediction_set == r2.prediction_set

    def test_save_creates_parent_dirs(self, fitted_calibrator):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "dir", "state.json")
            fitted_calibrator.save(path)
            assert os.path.exists(path)

    def test_save_unfitted_raises(self):
        cal = ConformalCalibrator()
        with pytest.raises(RuntimeError, match="unfitted"):
            cal.save("/tmp/nope.json")

    def test_saved_file_is_valid_json(self, fitted_calibrator):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            fitted_calibrator.save(path)
            with open(path) as f:
                data = json.load(f)
            assert "threshold" in data
            assert "coverage_target" in data
            assert "n_calibration" in data


# ═════════════════════════════════════════════════════════════════════════════
# HISTOGRAM CALIBRATOR
# ═════════════════════════════════════════════════════════════════════════════

class TestHistogramCalibrator:
    def test_fit_sets_state(self, fitted_histogram):
        assert fitted_histogram.is_fitted
        assert len(fitted_histogram.bins) == 10

    def test_bins_cover_unit_interval(self, fitted_histogram):
        bins = fitted_histogram.bins
        assert bins[0].bin_lower == pytest.approx(0.0)
        assert bins[-1].bin_upper == pytest.approx(1.0)

    def test_bins_are_contiguous(self, fitted_histogram):
        bins = fitted_histogram.bins
        for i in range(len(bins) - 1):
            assert bins[i].bin_upper == pytest.approx(bins[i + 1].bin_lower)

    def test_calibrate_returns_float(self, fitted_histogram):
        result = fitted_histogram.calibrate(0.5)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_calibrate_batch(self, fitted_histogram):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = fitted_histogram.calibrate_batch(scores)
        assert len(results) == 5
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_well_calibrated_data_has_low_ece(self):
        preds, labels = _make_well_calibrated_data(1000, seed=99)
        hc = HistogramCalibrator(n_bins=10)
        hc.fit(preds, labels)
        calibrated = hc.calibrate_batch(preds)
        ece_raw = expected_calibration_error(preds, labels)
        ece_cal = expected_calibration_error(calibrated, labels)
        # Histogram calibration should not make already-calibrated data worse
        assert ece_cal <= ece_raw + 0.05

    def test_fit_rejects_empty(self):
        hc = HistogramCalibrator()
        with pytest.raises(ValueError, match="empty"):
            hc.fit([], [])

    def test_fit_rejects_mismatched(self):
        hc = HistogramCalibrator()
        with pytest.raises(ValueError, match="same length"):
            hc.fit([0.5], [1, 0])

    def test_calibrate_before_fit_raises(self):
        hc = HistogramCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            hc.calibrate(0.5)

    def test_invalid_n_bins(self):
        with pytest.raises(ValueError):
            HistogramCalibrator(n_bins=0)

    def test_small_bin_falls_back_to_raw(self, cal_data):
        preds, labels = cal_data
        hc = HistogramCalibrator(n_bins=10, min_bin_count=9999)
        hc.fit(preds, labels)
        # All bins have < 9999 samples, so calibrate returns raw score
        assert hc.calibrate(0.5) == 0.5


class TestHistogramSaveLoad:
    def test_roundtrip(self, fitted_histogram):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hist.json")
            fitted_histogram.save(path)

            loaded = HistogramCalibrator()
            loaded.load(path)

            assert loaded.is_fitted
            assert loaded.n_bins == fitted_histogram.n_bins
            assert len(loaded.bins) == len(fitted_histogram.bins)

    def test_loaded_produces_same_results(self, fitted_histogram):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hist.json")
            fitted_histogram.save(path)

            loaded = HistogramCalibrator()
            loaded.load(path)

            for score in [0.05, 0.25, 0.55, 0.85, 0.95]:
                assert loaded.calibrate(score) == pytest.approx(
                    fitted_histogram.calibrate(score)
                )

    def test_save_unfitted_raises(self):
        hc = HistogramCalibrator()
        with pytest.raises(RuntimeError):
            hc.save("/tmp/nope.json")


# ═════════════════════════════════════════════════════════════════════════════
# CALIBRATION IMPROVES ECE
# ═════════════════════════════════════════════════════════════════════════════

class TestCalibrationImprovesECE:
    def test_histogram_reduces_ece(self):
        """Histogram calibration on miscalibrated data should lower ECE."""
        train_p, train_l = _make_calibration_data(800, seed=10)
        test_p, test_l = _make_calibration_data(400, seed=20)

        hc = HistogramCalibrator(n_bins=10)
        hc.fit(train_p, train_l)
        calibrated = hc.calibrate_batch(test_p)

        ece_before = expected_calibration_error(test_p, test_l)
        ece_after = expected_calibration_error(calibrated, test_l)

        assert ece_after < ece_before, (
            f"ECE should decrease: before={ece_before:.4f}, after={ece_after:.4f}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

class TestECE:
    def test_perfect_calibration_has_zero_ece(self):
        # Every bin has avg_pred == accuracy
        preds = [0.05] * 100 + [0.95] * 100
        labels = [0] * 95 + [1] * 5 + [1] * 95 + [0] * 5
        ece = expected_calibration_error(preds, labels, n_bins=10)
        assert ece < 0.05  # close to 0

    def test_worst_case_calibration(self):
        # Predict 1.0 when label is always 0
        preds = [1.0] * 100
        labels = [0] * 100
        ece = expected_calibration_error(preds, labels, n_bins=10)
        assert ece == pytest.approx(1.0)

    def test_ece_in_valid_range(self, cal_data):
        preds, labels = cal_data
        ece = expected_calibration_error(preds, labels)
        assert 0.0 <= ece <= 1.0

    def test_empty_input(self):
        assert expected_calibration_error([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [1, 0])


class TestBrierScore:
    def test_perfect_predictions(self):
        preds = [1.0, 0.0, 1.0, 0.0]
        labels = [1, 0, 1, 0]
        assert brier_score(preds, labels) == pytest.approx(0.0)

    def test_worst_predictions(self):
        preds = [1.0, 1.0]
        labels = [0, 0]
        assert brier_score(preds, labels) == pytest.approx(1.0)

    def test_brier_in_valid_range(self, cal_data):
        preds, labels = cal_data
        bs = brier_score(preds, labels)
        assert 0.0 <= bs <= 1.0

    def test_empty_input(self):
        assert brier_score([], []) == 0.0

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.5], [1, 0])


class TestCoverageRate:
    def test_full_coverage(self):
        sets = [{"A", "B"}, {"B", "C"}, {"A"}]
        labels = ["A", "B", "A"]
        assert coverage_rate(sets, labels) == pytest.approx(1.0)

    def test_zero_coverage(self):
        sets = [{"A"}, {"A"}]
        labels = ["B", "C"]
        assert coverage_rate(sets, labels) == pytest.approx(0.0)

    def test_partial_coverage(self):
        sets = [{"A"}, {"B"}, {"C"}, {"D"}]
        labels = ["A", "X", "C", "Y"]
        assert coverage_rate(sets, labels) == pytest.approx(0.5)

    def test_empty_input(self):
        assert coverage_rate([], []) == 0.0

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            coverage_rate([{"A"}], ["A", "B"])


class TestCalibrationCurve:
    def test_returns_two_lists(self):
        preds = [0.1, 0.5, 0.9]
        labels = [0, 1, 1]
        centers, accs = calibration_curve(preds, labels, n_bins=10)
        assert len(centers) == 10
        assert len(accs) == 10

    def test_centers_are_bin_midpoints(self):
        preds = [0.5]
        labels = [1]
        centers, _ = calibration_curve(preds, labels, n_bins=10)
        expected = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        for c, e in zip(centers, expected):
            assert c == pytest.approx(e, abs=0.01)

    def test_empty_bins_lie_on_diagonal(self):
        # Only bin around 0.5 has data
        preds = [0.55]
        labels = [1]
        centers, accs = calibration_curve(preds, labels, n_bins=10)
        # Bins with no data should have accuracy == center
        for i, (c, a) in enumerate(zip(centers, accs)):
            if i != 5:  # bin 5 is [0.5, 0.6)
                assert a == pytest.approx(c)

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            calibration_curve([0.5], [1, 0])


# ═════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_conformal_with_all_same_labels(self):
        preds = [0.5] * 100
        labels = [1] * 100
        cal = ConformalCalibrator(coverage_target=0.90)
        cal.fit(preds, labels)
        result = cal.calibrate(0.5)
        assert isinstance(result, CalibratedResult)

    def test_conformal_with_extreme_scores(self, fitted_calibrator):
        for score in [0.0, 1.0, -0.1, 1.5]:
            result = fitted_calibrator.calibrate(score)
            assert isinstance(result, CalibratedResult)

    def test_histogram_with_scores_at_boundaries(self, fitted_histogram):
        for score in [0.0, 0.1, 0.5, 1.0]:
            result = fitted_histogram.calibrate(score)
            assert 0.0 <= result <= 1.0

    def test_conformal_single_sample(self):
        cal = ConformalCalibrator(coverage_target=0.90)
        cal.fit([0.5], [1])
        result = cal.calibrate(0.5)
        assert isinstance(result, CalibratedResult)

    def test_histogram_single_sample(self):
        hc = HistogramCalibrator(n_bins=10, min_bin_count=1)
        hc.fit([0.5], [1])
        result = hc.calibrate(0.5)
        assert isinstance(result, float)

    def test_conformal_many_coverage_levels(self, cal_data):
        preds, labels = cal_data
        for target in [0.50, 0.80, 0.90, 0.95, 0.99]:
            cal = ConformalCalibrator(coverage_target=target)
            cal.fit(preds, labels)
            assert cal.is_fitted
            result = cal.calibrate(0.5)
            assert isinstance(result, CalibratedResult)
