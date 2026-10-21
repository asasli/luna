"""Multi-view novelty detection.

Scores computed on a single representation tend to be redundant: they are all functions of the
same embedding, so they flag largely the same samples. Complementary evidence can come from
*different representations* of the same input (e.g. two classifier backbones, physically
motivated features, a model of the raw signal). This module runs LUNA's scorers on several
representations ("views") and combines the views into ONE statistic that is calibrated at the
target false-alarm rate, using no anomalous data at all.

    from luna.multiview import ViewScorer, MultiViewDetector

    views = {"embedding": ViewScorer(), "physics": ViewScorer()}
    det = MultiViewDetector(views, combination="mean_rank", target_far=(0.01, 0.05))
    det.fit(train={"embedding": emb_train, "physics": phys_train},          # in-distribution only
            reference={"embedding": emb_ref, "physics": phys_ref},          # in-distribution only
            calibration={"embedding": emb_cal, "physics": phys_cal})        # in-distribution only
    out = det.predict({"embedding": emb_new, "physics": phys_new})
    out["flags"][0.01], out["statistic"], out["p_value"], out["view_p_values"]["physics"]
    det.overlap({"embedding": emb_x, "physics": phys_x})                   # how complementary are the views?

Each view's data is a LUNA data dict ({"embeddings": ..., optional "logits", "labels", ...}).
Three disjoint in-distribution splits are used: `train` fits the scorers, `reference` defines the
per-view rank / p-value scale, and `calibration` sets the combined threshold, so the reported
false-alarm rate is measured on data that played no other role.

If the view dicts carry `object_id` (the same array in every view), every view score is
aggregated per object (mean over the object's samples by default) and the reference scales, the
threshold and the predictions are all per object, as in `LUNAPipeline.calibrate`.

Combinations (each is ONE statistic, thresholded on `calibration` at the target rate; never OR
several independently calibrated thresholds, which inflates the false-alarm rate):
  "mean_rank"  mean of the per-view rank scores (default)
  "fisher"     Fisher's method: statistic = -2 * sum_v log p_v
  "union"      the most extreme view: statistic = -log(min_v p_v)
  "vote"       the k-th most extreme view: statistic = -log(k-th smallest p_v) (k=1 is "union")

On ASTRAnet spectra (two classifier backbones as two views, 107 rare objects, nominal per-object
1% false-alarm rate, ten calibration splits), "mean_rank" detected 41.8% of the rare objects,
"fisher" 41.0%, "vote" (k=2) 36.7% and "union" 34.9%; the most extreme of several views is
extreme more often for normal objects too, which raises the calibrated threshold of "union".
A single view on the concatenated embeddings of the two backbones detected 43.6%, so when the
representations can be concatenated, one view is the first thing to try. Measure on your own
data before choosing (see `MultiViewDetector.overlap`).
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from luna.pipeline import LUNAPipeline, OBJECT_KEY
from luna.thresholding import aggregate_by_object, calibrate_thresholds, p_values


def rank_pvalues(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Conformal-style p-value of each score against a reference sample (higher score = more anomalous):
    p = (1 + #{reference >= score}) / (n_reference + 1)."""
    return p_values(scores, reference)


def _object_ids(data: dict) -> np.ndarray | None:
    ids = data.get(OBJECT_KEY)
    return None if ids is None else np.asarray(ids)


class ViewScorer:
    """Zero-knowledge score for ONE representation: LUNA scorers fitted on in-distribution training
    data, each converted to a rank against an in-distribution reference split, aggregated by mean
    rank (and per object, by `object_aggregate`, when the data carry `object_id`).

    Args:
        scorers: LUNA scorer names to use (None = every scorer the data supports).
        fine_to_coarse: optional mapping for the hierarchical (HEC) scorer.
        scorer_kwargs: per-scorer keyword arguments.
        object_aggregate: "mean" (default) or "max" over the samples of an object.
    """

    def __init__(self, scorers: list[str] | None = None, fine_to_coarse: dict | None = None,
                 scorer_kwargs: dict | None = None, object_aggregate: str = "mean"):
        self.pipeline = LUNAPipeline(scorers=scorers, combiner="rank", fine_to_coarse=fine_to_coarse,
                                     scorer_kwargs=scorer_kwargs)
        self.object_aggregate = object_aggregate
        self._ref_components: dict[str, np.ndarray] = {}
        self._ref_aggregate: np.ndarray | None = None

    @property
    def methods(self) -> list[str]:
        return self.pipeline.active_methods

    def fit(self, train: dict, reference: dict) -> "ViewScorer":
        self.pipeline.fit(train)
        return self.set_reference(reference)

    def set_reference(self, reference: dict) -> "ViewScorer":
        """(Re)define the rank / p-value scale on an in-distribution reference split, without refitting.
        The component ranks are read per sample; the p-value scale is per object when the reference
        carries `object_id`."""
        comp = self.pipeline.score(reference)
        self._ref_components = {m: np.sort(np.asarray(comp[m], dtype=np.float64)) for m in self.methods}
        self._ref_aggregate = np.sort(self._unit(self._aggregate(comp), reference)[1])
        return self

    def _aggregate(self, comp: dict) -> np.ndarray:
        ranks = [np.searchsorted(self._ref_components[m], np.asarray(comp[m], dtype=np.float64)) / len(self._ref_components[m])
                 for m in self.methods]
        return np.mean(ranks, axis=0)

    def _unit(self, values: np.ndarray, data: dict) -> tuple[np.ndarray | None, np.ndarray]:
        ids = _object_ids(data)
        if ids is None:
            return None, values
        return aggregate_by_object(values, ids, self.object_aggregate)

    def component_scores(self, data: dict) -> dict[str, np.ndarray]:
        return self.pipeline.score(data)

    def score(self, data: dict) -> np.ndarray:
        """Mean rank of the view's scorers against the reference, in [0, 1]; higher = more anomalous.
        One value per object when the data carry `object_id` (sorted ids), else per sample."""
        return self._unit(self._aggregate(self.pipeline.score(data)), data)[1]

    def p_values(self, data: dict) -> np.ndarray:
        """p-value of the view score against the reference split."""
        return rank_pvalues(self._ref_aggregate, self.score(data))


class MultiViewDetector:
    """Combine several :class:`ViewScorer` views into one statistic, calibrated at the target
    false-alarm rate(s) on in-distribution data only.

    Args:
        views: mapping view name -> ViewScorer.
        combination: "mean_rank" (default), "fisher", "union" or "vote".
        k: for "vote", the number of views that must be extreme (default 1 = union).
        target_far: combined false-alarm rate(s) on the calibration split; the first is the main one.
        strict: infinite threshold when the calibration split is too small for a rate.
    """

    COMBINATIONS = ("mean_rank", "fisher", "union", "vote")

    def __init__(self, views: dict[str, ViewScorer], combination: str = "mean_rank", k: int = 1,
                 target_far: float | Sequence[float] = 0.01, strict: bool = True):
        if combination not in self.COMBINATIONS:
            raise ValueError(f"combination must be one of {self.COMBINATIONS}")
        if not views:
            raise ValueError("at least one view is required")
        self.views = dict(views)
        self.combination = combination
        self.k = int(k)
        fars = (target_far,) if np.isscalar(target_far) else tuple(target_far)
        self.fars = tuple(float(a) for a in fars)
        self.target_far = self.fars[0]
        self.strict = bool(strict)
        self.thresholds: dict[float, float] = {}
        self.threshold: float | None = None
        self.calibration_far: float | None = None
        self._cal_statistic: np.ndarray | None = None

    # ---- internals ----------------------------------------------------------------------------
    def _view_scores(self, data: dict) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
        ids = [_object_ids(data[v]) for v in self.views]
        if any((a is None) != (ids[0] is None) or (a is not None and not np.array_equal(a, ids[0])) for a in ids):
            raise ValueError("every view must carry the same object_id array (or none)")
        objects = None if ids[0] is None else np.unique(ids[0])
        return objects, {v: self.views[v].score(data[v]) for v in self.views}

    def _statistic(self, scores: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        p = {v: rank_pvalues(self.views[v]._ref_aggregate, scores[v]) for v in self.views}
        P = np.column_stack([p[v] for v in self.views])
        if self.combination == "mean_rank":
            stat = np.column_stack([scores[v] for v in self.views]).mean(axis=1)
        elif self.combination == "fisher":
            stat = -2.0 * np.log(P).sum(axis=1)
        elif self.combination == "union":
            stat = -np.log(P.min(axis=1))
        else:
            stat = -np.log(np.sort(P, axis=1)[:, self.k - 1])
        return stat, p

    # ---- API ----------------------------------------------------------------------------------
    def fit(self, train: dict, reference: dict, calibration: dict) -> "MultiViewDetector":
        """Fit every view on `train`, set per-view p-value scales on `reference`, and calibrate the
        combined threshold on `calibration` (all three in-distribution and disjoint)."""
        for v, view in self.views.items():
            view.pipeline.fit(train[v])
        return self.calibrate(reference, calibration)

    def calibrate(self, reference: dict, calibration: dict) -> "MultiViewDetector":
        """Re-set the per-view reference scales and the combined thresholds without refitting the
        scorers (e.g. to repeat the evaluation over several random in-distribution splits)."""
        for v, view in self.views.items():
            view.set_reference(reference[v])
        if self.combination == "vote" and not 1 <= self.k <= len(self.views):
            raise ValueError(f"k must be between 1 and the number of views ({len(self.views)})")
        _, scores = self._view_scores(calibration)
        stat, _ = self._statistic(scores)
        self._cal_statistic = np.sort(stat)
        self.thresholds = calibrate_thresholds(stat, self.fars, self.strict)
        self.threshold = self.thresholds[self.target_far]
        self.calibration_far = float((stat > self.threshold).mean())
        return self

    def predict(self, data: dict) -> dict:
        """Combined statistic, its conformal p-value, flags at every rate, and per-view scores and
        p-values (per object, with the sorted ids in "objects", when the data carry `object_id`)."""
        if self.threshold is None:
            raise RuntimeError("call fit() first")
        objects, scores = self._view_scores(data)
        stat, p = self._statistic(scores)
        flags = {a: stat > t for a, t in self.thresholds.items()}
        return {"objects": objects, "statistic": stat, "p_value": p_values(stat, self._cal_statistic),
                "flags": flags, "flagged": flags[self.target_far], "view_scores": scores,
                "view_p_values": p, "threshold": self.threshold, "thresholds": dict(self.thresholds)}

    def overlap(self, data: dict, level: float | None = None) -> dict:
        """How complementary are the views on `data`? For each pair of views: Jaccard index of the
        units each flags at its own p-value `level` (default the main target rate), and Spearman
        correlation of the view scores. Low values mean the views see different units."""
        level = self.target_far if level is None else level
        _, scores = self._view_scores(data)
        flags = {v: rank_pvalues(self.views[v]._ref_aggregate, scores[v]) <= level for v in self.views}
        out = {}
        for a, b in combinations(self.views, 2):
            inter, union = (flags[a] & flags[b]).sum(), (flags[a] | flags[b]).sum()
            out[f"{a}|{b}"] = {"jaccard": float(inter / union) if union else float("nan"),
                               "spearman": float(spearmanr(scores[a], scores[b])[0]),
                               "only_" + a: int((flags[a] & ~flags[b]).sum()),
                               "only_" + b: int((flags[b] & ~flags[a]).sum()), "both": int(inter)}
        return out
