"""Prediction sets for the known classes, stratified by novelty and confidence.

Split-conformal prediction sets with the true-label score s(x, y) = 1 - p_y(x). The threshold of
each calibration group is the ceil((n_g + 1)(1 - alpha))-th smallest score of that group, or +inf
when the group is too small (the class then enters every set of that group, which keeps the
guarantee). Mondrian calibration gives the coverage guarantee within every group, for
exchangeable in-distribution objects.

Overall (global) coverage can hide under-coverage in the objects where the classifier is least
reliable. On ASTRAnet spectra, two partly different groups were under-covered: objects with a high
LUNA anomaly score and objects with a low classifier confidence. Calibrating within the four cells
of these two binary variables (`joint_strata`) brought every examined group close to the target,
with larger sets. No coverage is claimed for classes outside the taxonomy.

    strata_cal = joint_strata(luna_cal, conf_cal, luna_cut, conf_cut)
    strata_new = joint_strata(luna_new, conf_new, luna_cut, conf_cut)
    sets = mondrian_sets(p_cal, y_cal, strata_cal, p_new, strata_new, alpha=0.1)
"""
from __future__ import annotations

import math

import numpy as np


def group_threshold(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold of one group: the ceil((n + 1)(1 - alpha))-th smallest score,
    +inf when n is too small for the level."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k - 1]) if 0 < k <= len(s) else float("inf")


def mondrian_sets(
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
    strata_cal: np.ndarray | None,
    probs_new: np.ndarray,
    strata_new: np.ndarray | None,
    alpha: float = 0.1,
) -> np.ndarray:
    """Prediction sets calibrated within each stratum (one global group when strata are None).

    Args:
        probs_cal: (n, C) class probabilities of the calibration objects.
        y_cal: (n,) true labels of the calibration objects.
        strata_cal: (n,) stratum of each calibration object, or None.
        probs_new: (m, C) class probabilities of the new objects.
        strata_new: (m,) stratum of each new object, or None.
        alpha: miscoverage level (0.1 for 90% sets).

    Returns:
        (m, C) boolean membership matrix; a row may be empty.
    """
    probs_cal, probs_new = np.asarray(probs_cal, float), np.asarray(probs_new, float)
    y_cal = np.asarray(y_cal)
    s_cal = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]
    if strata_cal is None:
        strata_cal = np.zeros(len(y_cal), int)
        strata_new = np.zeros(len(probs_new), int)
    strata_cal, strata_new = np.asarray(strata_cal), np.asarray(strata_new)
    q = {g: group_threshold(s_cal[strata_cal == g], alpha) for g in np.unique(strata_new)}
    qn = np.array([q[g] for g in strata_new])
    return (1.0 - probs_new) <= qn[:, None]


def joint_strata(anomaly: np.ndarray, confidence: np.ndarray, anomaly_cut: float, confidence_cut: float) -> np.ndarray:
    """Four cells of (high anomaly score) x (low confidence): 0 neither, 1 low confidence only,
    2 high anomaly only, 3 both. Fix both cuts on separate in-distribution data (e.g. the 80th
    percentile of the anomaly score and of 1 - max probability on validation objects)."""
    high = np.asarray(anomaly) > anomaly_cut
    low_conf = (1.0 - np.asarray(confidence)) > confidence_cut
    return 2 * high.astype(int) + low_conf.astype(int)


def coverage(sets: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """True-label coverage, mean set size and empty-set rate of `sets` (optionally on a subgroup)."""
    sets, y = np.asarray(sets, bool), np.asarray(y)
    if mask is not None:
        sets, y = sets[mask], y[mask]
    if len(y) == 0:
        return {"n": 0, "coverage": float("nan"), "mean_size": float("nan"), "empty": float("nan")}
    return {"n": int(len(y)), "coverage": float(sets[np.arange(len(y)), y].mean()),
            "mean_size": float(sets.sum(1).mean()), "empty": float((sets.sum(1) == 0).mean())}
