"""FAR-based threshold calibration and conformal p-values for anomaly detection.

The calibration scores must come from in-distribution objects that were used neither to fit
the scorers nor to fit the combiner; otherwise the realised false-alarm rate on new data is
higher than the target.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def calibrate_threshold(id_scores: np.ndarray, target_far: float = 0.01) -> float:
    """Threshold above which a fraction `target_far` of the calibration scores lie.

    Conservative split-conformal choice: the ceil((n + 1)(1 - far))-th smallest calibration
    score, so that for exchangeable in-distribution objects the probability of a false alarm
    is at most `target_far`. When n is too small for the requested rate (n + 1 < 1 / far),
    the largest calibration score is returned and the realised rate can exceed the target.

    Args:
        id_scores: anomaly scores of the held-out in-distribution calibration objects.
        target_far: desired false-alarm rate (default 0.01 = 1%).

    Returns:
        Threshold value; an object is flagged when its score is strictly larger.
    """
    s = np.sort(np.asarray(id_scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    k = int(np.ceil((n + 1) * (1.0 - target_far)))
    return float(s[min(max(k, 1), n) - 1])


def calibrate_thresholds(id_scores: np.ndarray, fars: Iterable[float]) -> dict[float, float]:
    """`calibrate_threshold` for several false-alarm rates at once."""
    return {float(a): calibrate_threshold(id_scores, a) for a in fars}


def p_values(scores: np.ndarray, id_scores: np.ndarray) -> np.ndarray:
    """Conformal p-values: (1 + #{calibration scores >= s}) / (n + 1).

    Small values are anomalous. Flagging p <= far is equivalent to the threshold of
    `calibrate_threshold` at that rate, so one calibration serves every operating point.
    """
    s = np.sort(np.asarray(id_scores, dtype=float))
    n = len(s)
    ge = n - np.searchsorted(s, np.asarray(scores, dtype=float), side="left")
    return (1.0 + ge) / (n + 1.0)


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Flag samples exceeding the threshold.

    Args:
        scores: anomaly scores.
        threshold: detection threshold.

    Returns:
        Boolean array, True for flagged (anomalous) samples.
    """
    return scores > threshold
