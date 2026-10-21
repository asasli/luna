"""FAR-based threshold calibration, conformal p-values, object aggregation and FDR control.

The calibration scores must come from in-distribution objects that were used neither to fit
the scorers nor to fit the combiner; otherwise the realised false-alarm rate on new data is
higher than the target. Calibrate in the unit you report: when an object has several samples
(e.g. several spectra or epochs), aggregate the samples of each object first and calibrate on
the object scores, since a per-sample threshold flags an object with many samples more often.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def calibrate_threshold(id_scores: np.ndarray, target_far: float = 0.01, strict: bool = True) -> float:
    """Threshold above which a fraction `target_far` of the calibration scores lie.

    Conservative split-conformal choice: the ceil((n + 1)(1 - far))-th smallest calibration
    score, so that for exchangeable in-distribution objects the probability of a false alarm
    is at most `target_far`. When n is too small for the requested rate (n + 1 < 1 / far), the
    threshold is +inf (nothing is flagged), which keeps the guarantee; `strict=False` returns the
    largest calibration score instead, and the realised rate can then exceed the target.

    Args:
        id_scores: anomaly scores of the held-out in-distribution calibration objects.
        target_far: desired false-alarm rate (default 0.01 = 1%).
        strict: return +inf when the rate cannot be guaranteed with n calibration scores (default).

    Returns:
        Threshold value; an object is flagged when its score is strictly larger.
    """
    s = np.sort(np.asarray(id_scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    k = int(np.ceil((n + 1) * (1.0 - target_far)))
    if k > n and strict:
        return float("inf")
    return float(s[min(max(k, 1), n) - 1])


def calibrate_thresholds(id_scores: np.ndarray, fars: Iterable[float], strict: bool = True) -> dict[float, float]:
    """`calibrate_threshold` for several false-alarm rates at once."""
    return {float(a): calibrate_threshold(id_scores, a, strict) for a in fars}


def p_values(scores: np.ndarray, id_scores: np.ndarray) -> np.ndarray:
    """Conformal p-values: (1 + #{calibration scores >= s}) / (n + 1).

    Small values are anomalous. Flagging p <= far is equivalent to the threshold of
    `calibrate_threshold` at that rate, so one calibration serves every operating point.
    """
    s = np.sort(np.asarray(id_scores, dtype=float))
    n = len(s)
    ge = n - np.searchsorted(s, np.asarray(scores, dtype=float), side="left")
    return (1.0 + ge) / (n + 1.0)


def aggregate_by_object(scores: np.ndarray, object_ids: np.ndarray, how: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-sample scores to one score per object.

    Args:
        scores: (N,) per-sample scores (larger = more anomalous).
        object_ids: (N,) object identifier of each sample.
        how: "mean" (default) or "max" (an object is as anomalous as its most anomalous sample).
            On ASTRAnet spectra the mean detected more rare objects than the max at the same
            per-object false-alarm rate, for each of three classifier backbones, because the
            maximum over many epochs also makes multi-epoch normal objects extreme.

    Returns:
        (objects, object_scores), with the objects in sorted order.
    """
    objects, inverse = np.unique(np.asarray(object_ids), return_inverse=True)
    scores = np.asarray(scores, dtype=float)
    if how == "max":
        out = np.full(len(objects), -np.inf)
        np.maximum.at(out, inverse, scores)
    elif how == "mean":
        out = np.bincount(inverse, weights=scores, minlength=len(objects)) / np.bincount(inverse, minlength=len(objects))
    else:
        raise ValueError("how must be 'max' or 'mean'")
    return objects, out


def conformal_bh(p: np.ndarray, q: float = 0.1) -> np.ndarray:
    """Benjamini-Hochberg selection on conformal p-values (Bates et al. 2023).

    Flags the objects whose p-values pass the BH step-up rule at level `q`. For exchangeable
    in-distribution objects, the expected fraction of in-distribution objects among the flagged
    ones (the false-discovery rate) is at most q. The rule depends on the whole batch, so a flag
    is a property of the batch and not of the object alone.

    Returns:
        Boolean array of flags.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    passed = np.flatnonzero(p[order] <= q * np.arange(1, n + 1) / n)
    flags = np.zeros(n, dtype=bool)
    if len(passed):
        flags[order[: passed.max() + 1]] = True
    return flags


def storey_pi0(p: np.ndarray, lam: float = 0.5) -> float:
    """Storey's estimate of the in-distribution fraction of a batch from its conformal p-values."""
    p = np.asarray(p, dtype=float)
    return float(min(1.0, (p > lam).sum() / ((1.0 - lam) * len(p))))


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a threshold to anomaly scores.

    Args:
        scores: anomaly scores.
        threshold: detection threshold.

    Returns:
        Boolean array, True for flagged (anomalous) samples.
    """
    return scores > threshold
