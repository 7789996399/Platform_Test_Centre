"""
Calibration metrics for evaluating Layer 3 calibrators.

All functions accept plain Python lists — no numpy required.
"""

import math
from typing import List, Set, Tuple


def expected_calibration_error(
    predictions: List[float],
    labels: List[int],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Partitions [0, 1] into *n_bins* equal-width bins. For each bin,
    computes |avg_confidence − accuracy| weighted by bin count.

    Parameters
    ----------
    predictions : list[float]
        Predicted probabilities (0.0 – 1.0).
    labels : list[int]
        Binary ground truth (0 or 1).
    n_bins : int
        Number of bins.

    Returns
    -------
    float
        ECE value in [0, 1].  Lower is better.
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    n = len(predictions)
    if n == 0:
        return 0.0

    width = 1.0 / n_bins
    ece = 0.0

    for i in range(n_bins):
        lo = i * width
        hi = (i + 1) * width
        in_bin = [
            (p, l)
            for p, l in zip(predictions, labels)
            if (lo <= p < hi) or (i == n_bins - 1 and p == hi)
        ]
        if not in_bin:
            continue
        avg_conf = sum(p for p, _ in in_bin) / len(in_bin)
        accuracy = sum(l for _, l in in_bin) / len(in_bin)
        ece += len(in_bin) / n * abs(avg_conf - accuracy)

    return ece


def brier_score(
    predictions: List[float],
    labels: List[int],
) -> float:
    """
    Brier score (mean squared error of probability estimates).

    Parameters
    ----------
    predictions : list[float]
        Predicted probabilities (0.0 – 1.0).
    labels : list[int]
        Binary ground truth (0 or 1).

    Returns
    -------
    float
        Brier score in [0, 1].  Lower is better.
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    n = len(predictions)
    if n == 0:
        return 0.0
    return sum((p - l) ** 2 for p, l in zip(predictions, labels)) / n


def coverage_rate(
    prediction_sets: List[Set[str]],
    true_labels: List[str],
) -> float:
    """
    Empirical coverage rate of prediction sets.

    Returns the fraction of instances where the true label is contained
    in the prediction set.

    Parameters
    ----------
    prediction_sets : list[set[str]]
        Prediction sets (e.g. {{"BRIEF", "STANDARD"}}).
    true_labels : list[str]
        True review-level labels.

    Returns
    -------
    float
        Coverage rate in [0, 1].
    """
    if len(prediction_sets) != len(true_labels):
        raise ValueError("prediction_sets and true_labels must have the same length")
    n = len(prediction_sets)
    if n == 0:
        return 0.0
    covered = sum(1 for ps, lbl in zip(prediction_sets, true_labels) if lbl in ps)
    return covered / n


def calibration_curve(
    predictions: List[float],
    labels: List[int],
    n_bins: int = 10,
) -> Tuple[List[float], List[float]]:
    """
    Compute calibration curve (reliability diagram data).

    Parameters
    ----------
    predictions : list[float]
        Predicted probabilities.
    labels : list[int]
        Binary ground truth.
    n_bins : int
        Number of bins.

    Returns
    -------
    (bin_centers, bin_accuracies)
        Lists of length *n_bins*.  Bins with no samples have
        accuracy equal to the bin center (diagonal).
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")

    width = 1.0 / n_bins
    centers: List[float] = []
    accuracies: List[float] = []

    for i in range(n_bins):
        lo = i * width
        hi = (i + 1) * width
        center = (lo + hi) / 2
        in_bin = [
            l for p, l in zip(predictions, labels)
            if (lo <= p < hi) or (i == n_bins - 1 and p == hi)
        ]
        if in_bin:
            acc = sum(in_bin) / len(in_bin)
        else:
            acc = center  # no data → lie on diagonal
        centers.append(round(center, 6))
        accuracies.append(round(acc, 6))

    return centers, accuracies
