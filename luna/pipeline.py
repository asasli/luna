"""LUNAPipeline — orchestrates scorers, combiner, thresholding, and reporting."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from luna.scorers import SCORER_REGISTRY, get_available_scorers, get_scorer_families
from luna.combiners import get_combiner, BaseCombiner, MeanRankCombiner
from luna.thresholding import calibrate_threshold, calibrate_thresholds, apply_threshold, p_values


def _take(data: dict, idx: np.ndarray) -> dict:
    """Rows `idx` of every array in a data dict (None values are kept)."""
    return {k: (None if v is None else np.asarray(v)[idx]) for k, v in data.items()}


def _n_rows(data: dict) -> int:
    return len(next(np.asarray(v) for v in data.values() if v is not None))


class LUNAPipeline:
    """End-to-end anomaly detection pipeline.

    Orchestrates:
      1. Scorer fitting on training data
      2. OOD score computation
      3. Combiner training on ID vs OOD scores
      4. Threshold calibration at a target false-alarm rate
      5. Full prediction (scores + ensemble + flags)
      6. Report + visualization generation

    Example (with labelled outliers):
        pipe = LUNAPipeline(combiner="lightgbm", target_far=(0.01, 0.05, 0.10))
        pipe.fit(train_data)
        pipe.fit_combiner(val_data, rare_data)          # half of val_data calibrates
        result = pipe.predict(new_data)                 # p_value, flags at 1%, 5%, 10%

    Example (blind, no labelled outlier):
        pipe = LUNAPipeline(target_far=(0.01, 0.05, 0.10)).fit(train_data)
        pipe.fit_blind(val_data)    # mean percentile rank, frozen reference ranks
        table = pipe.evaluate(test_normals, rare_data, groups=rare_families)
    """

    def __init__(
        self,
        scorers: list[str] | None = None,
        combiner: str | BaseCombiner = "lightgbm",
        target_far: float | Sequence[float] = 0.01,
        fine_to_coarse: dict[int, int] | None = None,
        scorer_kwargs: dict[str, dict] | None = None,
        combiner_kwargs: dict | None = None,
    ):
        """
        Args:
            scorers: list of scorer names to use, or None for auto-detection.
            combiner: combiner name or BaseCombiner instance.
            target_far: false-alarm rate(s) for threshold calibration (default 1%). A
                sequence, e.g. (0.01, 0.05, 0.10), calibrates one threshold per rate; the
                first entry is the main operating point (`flagged`).
            fine_to_coarse: mapping fine class idx -> coarse class idx (for HEC).
            scorer_kwargs: per-scorer kwargs, e.g. {"odin": {"temperature": 2.0}}.
            combiner_kwargs: kwargs passed to the combiner constructor.
        """
        self._requested_scorers = scorers
        self._combiner_spec = combiner
        fars = (target_far,) if np.isscalar(target_far) else tuple(target_far)
        self.fars = tuple(float(a) for a in fars)
        self.target_far = self.fars[0]
        self.fine_to_coarse = fine_to_coarse
        self._scorer_kwargs = scorer_kwargs or {}
        self._combiner_kwargs = combiner_kwargs or {}

        self._scorers: dict[str, object] = {}
        self._combiner: BaseCombiner | None = None
        self._threshold: float | None = None
        self._thresholds: dict[float, float] = {}
        self._cal_scores: np.ndarray | None = None
        self._cal_method_scores: dict[str, np.ndarray] | None = None
        self._active_methods: list[str] = []
        self._fitted = False
        self._combiner_fitted = False

    @property
    def active_methods(self) -> list[str]:
        """Names of scorers that are currently active."""
        return list(self._active_methods)

    @property
    def families(self) -> dict[str, list[str]]:
        """Active scorers grouped by family."""
        fam = {}
        for name in self._active_methods:
            family = self._scorers[name].family
            fam.setdefault(family, []).append(name)
        return fam

    def fit(self, train_data: dict[str, np.ndarray]) -> LUNAPipeline:
        """Fit scorer statistics on training data.

        Args:
            train_data: dict with keys like "embeddings", "logits", "labels", etc.
                "embeddings" and "labels" are required.

        Returns:
            self
        """
        data_keys = {k for k, v in train_data.items() if v is not None}

        # Determine which scorers to use
        if self._requested_scorers is not None:
            method_names = self._requested_scorers
        else:
            method_names = get_available_scorers(data_keys)

        # Instantiate and fit scorers
        self._scorers = {}
        self._active_methods = []
        for name in method_names:
            if name not in SCORER_REGISTRY:
                raise ValueError(
                    f"Unknown scorer '{name}'. Available: {list(SCORER_REGISTRY.keys())}"
                )
            cls = SCORER_REGISTRY[name]
            if not cls.requires.issubset(data_keys):
                missing = cls.requires - data_keys
                print(f"  [LUNA] Skipping '{name}' — missing: {missing}")
                continue

            kwargs = self._scorer_kwargs.get(name, {})
            if name == "hec" and self.fine_to_coarse is not None:
                kwargs.setdefault("fine_to_coarse", self.fine_to_coarse)

            scorer = cls(**kwargs)
            import time as _t
            _t0 = _t.time()
            scorer.fit(train_data)
            print(f"  [LUNA] fit {name} in {_t.time()-_t0:.1f}s", flush=True)
            self._scorers[name] = scorer
            self._active_methods.append(name)

        self._fitted = True
        print(f"  [LUNA] Fitted {len(self._active_methods)} scorers: {self._active_methods}")
        return self

    def score(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute all active OOD scores on new data.

        Args:
            data: dict with the same keys used during fit (labels not required).

        Returns:
            Dict mapping scorer name -> (N,) score array.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .score().")

        # NOTE: no id()-based memoization — Python recycles array memory ids, so a
        # fresh temporary array can collide with a GC'd one and return wrong cached
        # scores. Always recompute (fast with brute-kNN + shrinkage-cov scorers).
        scores = {}
        for name in self._active_methods:
            scores[name] = self._scorers[name].score(data)
        return scores

    def _scores_to_matrix(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Stack score dict into (N, M) matrix in canonical order."""
        return np.column_stack([scores[m] for m in self._active_methods])

    def fit_combiner(
        self,
        id_data: dict[str, np.ndarray],
        ood_data: dict[str, np.ndarray],
        calibration_data: dict[str, np.ndarray] | None = None,
        cal_fraction: float = 0.5,
        seed: int = 0,
        verbose: bool = True,
    ) -> LUNAPipeline:
        """Fit the combiner on ID (validation) vs OOD (rare/anomaly) data, then calibrate.

        Args:
            id_data: in-distribution data.
            ood_data: out-of-distribution / anomaly data.
            calibration_data: held-out in-distribution data for the thresholds. When None,
                a random fraction `cal_fraction` of `id_data` is set aside for calibration
                and the combiner is fitted on the rest, so the thresholds are never set on
                objects the combiner has seen.
            cal_fraction: fraction of `id_data` used for calibration when no
                `calibration_data` is given.
            seed: random seed of that split.
            verbose: print AUROC and threshold info.

        Returns:
            self
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .fit_combiner().")
        if calibration_data is None:
            id_data, calibration_data = self._split(id_data, cal_fraction, seed)

        X_id = self._scores_to_matrix(self.score(id_data))
        X_ood = self._scores_to_matrix(self.score(ood_data))
        X = np.vstack([X_id, X_ood])
        y = np.concatenate([
            np.zeros(X_id.shape[0], dtype=int),
            np.ones(X_ood.shape[0], dtype=int),
        ])

        self._combiner = get_combiner(self._combiner_spec, **self._combiner_kwargs)
        if verbose:
            cv_aurocs = self._combiner.cross_val_auroc(X, y)
            print(f"  [LUNA] Combiner CV AUROC = {cv_aurocs.mean():.4f} "
                  f"+/- {cv_aurocs.std():.4f}")
        self._combiner.fit(X, y)
        self._combiner_fitted = True
        return self.calibrate(calibration_data, verbose=verbose)

    def fit_blind(
        self,
        reference_data: dict[str, np.ndarray],
        calibration_data: dict[str, np.ndarray] | None = None,
        cal_fraction: float = 0.5,
        seed: int = 0,
        verbose: bool = True,
    ) -> LUNAPipeline:
        """Blind consensus: mean percentile rank across the scorers, no labelled anomaly.

        The percentile of each scorer is read from the reference in-distribution objects and
        frozen, so the consensus of an object does not depend on the other objects scored
        with it. The thresholds are set on separate calibration objects.

        Args:
            reference_data: in-distribution data that defines the percentile ranks.
            calibration_data: held-out in-distribution data for the thresholds. When None,
                a random fraction `cal_fraction` of `reference_data` is used.
            cal_fraction, seed: as in `fit_combiner`.
            verbose: print threshold info.

        Returns:
            self
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .fit_blind().")
        if calibration_data is None:
            reference_data, calibration_data = self._split(reference_data, cal_fraction, seed)
        X_ref = self._scores_to_matrix(self.score(reference_data))
        self._combiner = MeanRankCombiner().fit(X_ref, np.zeros(len(X_ref), dtype=int))
        self._combiner_fitted = True
        return self.calibrate(calibration_data, verbose=verbose)

    def calibrate(
        self, calibration_data: dict[str, np.ndarray], verbose: bool = True,
    ) -> LUNAPipeline:
        """Set the thresholds and the p-value reference on held-out in-distribution data.

        Can be called again on new normals (e.g. a new survey season) without refitting.
        """
        if not self._combiner_fitted:
            raise RuntimeError("Fit a combiner (fit_combiner or fit_blind) first.")
        cal_scores = self.score(calibration_data)
        cal = self._combiner.predict(self._scores_to_matrix(cal_scores))
        self._cal_scores = np.sort(cal)
        self._cal_method_scores = {m: np.sort(v) for m, v in cal_scores.items()}
        self._thresholds = calibrate_thresholds(cal, self.fars)
        self._threshold = self._thresholds[self.target_far]
        if verbose:
            n = len(cal)
            for a, thr in self._thresholds.items():
                small = " (too few calibration objects for this rate)" if (n + 1) * a < 1 else ""
                print(f"  [LUNA] Threshold @ FAR={a:.1%}: {thr:.4f} on {n} calibration "
                      f"objects{small}")
        return self

    @staticmethod
    def _split(data: dict, fraction: float, seed: int) -> tuple[dict, dict]:
        n = _n_rows(data)
        perm = np.random.default_rng(seed).permutation(n)
        n_cal = int(round(fraction * n))
        if n_cal < 1 or n_cal >= n:
            raise ValueError(f"cannot split {n} objects with cal_fraction={fraction}")
        return _take(data, perm[n_cal:]), _take(data, perm[:n_cal])

    @property
    def thresholds(self) -> dict[float, float]:
        """Calibrated threshold per false-alarm rate."""
        return dict(self._thresholds)

    def predict(self, data: dict[str, np.ndarray]) -> dict:
        """Full prediction: scores, ensemble, conformal p-value and flags at every rate.

        Args:
            data: input data dict.

        Returns:
            Dict with:
                "scores": dict of per-scorer arrays
                "ensemble_score": (N,) combined anomaly score
                "p_value": (N,) conformal p-value against the calibration objects
                "flagged": (N,) boolean array at the main rate (first `target_far`)
                "flags": {far: (N,) boolean array} for every calibrated rate
                "threshold": float, main-rate threshold
                "thresholds": {far: threshold}
        """
        if not self._combiner_fitted or self._cal_scores is None:
            raise RuntimeError("Call .fit_combiner() or .fit_blind() before .predict().")

        scores = self.score(data)
        ens = self._combiner.predict(self._scores_to_matrix(scores))
        flags = {a: apply_threshold(ens, thr) for a, thr in self._thresholds.items()}
        return {
            "scores": scores,
            "ensemble_score": ens,
            "p_value": p_values(ens, self._cal_scores),
            "flagged": flags[self.target_far],
            "flags": flags,
            "threshold": self._threshold,
            "thresholds": dict(self._thresholds),
        }

    def evaluate(
        self,
        id_data: dict[str, np.ndarray],
        ood_data: dict[str, np.ndarray],
        groups: Sequence[str] | None = None,
        methods: bool = False,
    ) -> pd.DataFrame:
        """Detection table at every calibrated rate.

        `id_data` must be independent in-distribution test objects (not the reference or
        calibration objects), so that the realised false-alarm rate is an honest estimate.

        Args:
            id_data: independent in-distribution test objects.
            ood_data: anomalies.
            groups: optional label per anomaly (e.g. its family), for per-group rows.
            methods: also report every single scorer, thresholded on the same calibration
                objects (requires the calibration scores, kept by `calibrate`).

        Returns:
            DataFrame with one row per (rule, rate, group): target FAR, realised FPR on
            `id_data`, AUROC, number detected and detection rate.
        """
        pi, po = self.predict(id_data), self.predict(ood_data)
        rules = {"ensemble": (pi["ensemble_score"], po["ensemble_score"], self._thresholds)}
        if methods:
            if getattr(self, "_cal_method_scores", None) is None:
                raise RuntimeError("per-method thresholds need calibrate() from v0.3")
            for m in self._active_methods:
                rules[m] = (pi["scores"][m], po["scores"][m],
                            calibrate_thresholds(self._cal_method_scores[m], self.fars))
        groups = None if groups is None else np.asarray(groups)
        sel = {"all": np.ones(len(po["ensemble_score"]), bool)}
        if groups is not None:
            sel.update({g: groups == g for g in sorted(set(groups))})
        rows = []
        for rule, (si, so, thr) in rules.items():
            for a, t in thr.items():
                fpr = float((si > t).mean())
                for g, mask in sel.items():
                    n = int(mask.sum())
                    y = np.r_[np.zeros(len(si)), np.ones(n)]
                    rows.append(dict(rule=rule, target_far=a, realised_fpr=fpr, group=g, n=n,
                                     auroc=roc_auc_score(y, np.r_[si, so[mask]]),
                                     detected=int((so[mask] > t).sum()),
                                     det_rate=float((so[mask] > t).mean())))
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        """Pickle the fitted pipeline (scorers, combiner, calibration)."""
        from luna import __version__
        with open(path, "wb") as f:
            pickle.dump(dict(luna_version=__version__, pipeline=self), f)

    @staticmethod
    def load(path: str | Path) -> LUNAPipeline:
        """Load a pipeline written by `save`. Only load files you trust (pickle)."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj["pipeline"]

    def family_analysis(
        self,
        id_data: dict[str, np.ndarray],
        ood_data: dict[str, np.ndarray],
        seed: int = 0,
    ) -> dict:
        """Per-family AUROC and detection analysis.

        Within each scorer family, the scores are combined as the mean percentile rank. The
        ranks are read from one half of `id_data` and frozen, the thresholds are set on the
        other half, and the AUROC compares that second half with the anomalies, so the result
        for an anomaly does not depend on which other anomalies are scored with it.

        Args:
            id_data: in-distribution data (not used to fit the scorers).
            ood_data: out-of-distribution data.
            seed: random seed of the reference/calibration split of `id_data`.

        Returns:
            Dict with per-family AUROC, detection counts and flags at the main rate, and the
            detection rate at every calibrated rate ("det_rate_by_far").
        """
        ref_data, cal_data = self._split(id_data, 0.5, seed)
        ref, cal, ood = (self.score(d) for d in (ref_data, cal_data, ood_data))
        n_cal = len(next(iter(cal.values())))
        n_ood = len(next(iter(ood.values())))
        y = np.concatenate([np.zeros(n_cal), np.ones(n_ood)])

        result = {}
        for fam, methods in self.families.items():
            comb = MeanRankCombiner().fit(np.column_stack([ref[m] for m in methods]),
                                          np.zeros(len(ref[methods[0]]), dtype=int))
            rank_cal = comb.predict(np.column_stack([cal[m] for m in methods]))
            rank_ood = comb.predict(np.column_stack([ood[m] for m in methods]))
            thr = calibrate_thresholds(rank_cal, self.fars)
            flags = rank_ood > thr[self.target_far]
            result[fam] = {
                "auroc": roc_auc_score(y, np.concatenate([rank_cal, rank_ood])),
                "detected": int(flags.sum()),
                "total": n_ood,
                "det_rate": flags.mean(),
                "flags": flags,
                "det_rate_by_far": {a: float((rank_ood > v).mean()) for a, v in thr.items()},
            }
        return result

    def report(
        self,
        id_data: dict[str, np.ndarray],
        ood_data: dict[str, np.ndarray],
        out_dir: str | Path,
        class_names: list[str] | None = None,
        meta: list[dict] | None = None,
    ) -> None:
        """Generate full report: CSV tables, LaTeX, and figures.

        Args:
            id_data: in-distribution data.
            ood_data: out-of-distribution data.
            out_dir: output directory.
            class_names: list of class name strings.
            meta: per-sample metadata dicts for the OOD data
                  (keys like "obj_id", "class").
        """
        from luna.report import (
            generate_anomaly_report,
            generate_category_ablation,
            write_latex_table,
        )
        from luna.viz import (
            plot_detection_map,
            plot_complementarity_map,
        )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compute scores and predictions
        pred = self.predict(ood_data)
        fam_analysis = self.family_analysis(id_data, ood_data)

        # Per-method flags for detection map
        ood_scores = pred["scores"]
        method_flags = {}
        for m in self._active_methods:      # single scorers, thresholded on the calibration objects
            thr = calibrate_threshold(self._cal_method_scores[m], self.target_far)
            method_flags[m] = ood_scores[m] > thr

        # Family flags
        fam_flags = {f: info["flags"] for f, info in fam_analysis.items()}

        # True classes from meta
        if meta is not None:
            true_classes = np.array([m.get("class", "?") for m in meta])
        else:
            true_classes = None

        # Detection at every calibrated rate, with the realised FPR on id_data
        self.evaluate(id_data, ood_data, groups=true_classes, methods=True).to_csv(
            out_dir / "detection_by_far.csv", index=False)

        # Anomaly report CSV
        report_df = generate_anomaly_report(
            scores=ood_scores,
            ensemble_score=pred["ensemble_score"],
            flags=pred["flagged"],
            p_value=pred["p_value"],
            meta=meta,
            class_names=class_names,
            logits=ood_data.get("logits"),
            family_flags=fam_flags,
        )
        report_df.to_csv(out_dir / "anomaly_report.csv", index=False)

        # Category ablation
        ablation_df = generate_category_ablation(fam_analysis)
        ablation_df.to_csv(out_dir / "category_ablation.csv", index=False)

        # LaTeX table
        write_latex_table(report_df, out_dir / "anomaly_report_published.tex")

        # Detection map
        n_ood = len(pred["ensemble_score"])
        plot_detection_map(
            method_flags=method_flags,
            ensemble_flags=pred["flagged"],
            true_classes=true_classes,
            family_flags=fam_flags,
            out_path=out_dir / "detection_map",
            target_far=self.target_far,
        )

        # Complementarity map
        if true_classes is not None:
            plot_complementarity_map(
                family_flags=fam_flags,
                true_classes=true_classes,
                out_path=out_dir / "complementarity_map",
            )

        print(f"  [LUNA] Report written to {out_dir}")
