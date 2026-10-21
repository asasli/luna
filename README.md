# LUNA — Library for Uncertainty and Novelty Analysis

Classifier-agnostic **out-of-distribution / novelty detection** for any model that
produces embeddings and logits. LUNA computes **16 anomaly scores in four families**
— uncertainty (MSP, entropy, energy, ODIN), distance (Mahalanobis global/class/within,
kNN, cosine, typicality), density (GMM, isolation forest, LOF, PCA-reconstruction),
hybrid (HEC, MRS) — and combines them with a trainable ensemble (LightGBM by default),
thresholded at a user-chosen false-alarm rate.

No assumption is made about the upstream model: anything that yields a
per-sample embedding vector (and optionally logits / projection heads) works —
CNNs, transformers, autoencoders, in any domain.

## Install

```bash
pip install git+https://github.com/asasli/LUNA          # library
pip install "luna-ood[viz] @ git+https://github.com/asasli/LUNA"   # + UMAP/plotly extras
# or for development:
git clone https://github.com/asasli/LUNA && pip install -e ./LUNA
```

## Quick start

```python
import numpy as np
from luna import LUNAPipeline

# Your model's outputs, as plain arrays:
train = {"embeddings": Ztr, "logits": Ltr, "labels": ytr}   # in-distribution train
val   = {"embeddings": Zva, "logits": Lva, "labels": yva}   # in-distribution holdout
query = {"embeddings": Zq,  "logits": Lq}                   # anything you want to score

pipe = LUNAPipeline(target_far=(0.01, 0.05, 0.10))
pipe.fit(train)                       # fits the 16 scorers on ID statistics

scores = pipe.score(query)            # dict: {"msp": ..., "mahal_within": ..., ...}

# Blind consensus, no labelled outlier needed: mean percentile rank across the scorers,
# with the ranks frozen on half of `val` and the thresholds set on the other half
pipe.fit_blind(val)
out = pipe.predict(query)             # ensemble_score, p_value, flags at 1%, 5% and 10%

# Or a supervised combiner, if you have a small set of known outliers
# (half of `val` is kept aside for the thresholds; pass calibration_data= to choose it)
pipe.fit_combiner(val, known_outliers)

# Detection and realised false-alarm rate at every rate, on independent test normals
table = pipe.evaluate(test_normals, outliers, groups=outlier_families, methods=True)
```

`p_value` is the conformal p-value of each object against the calibration normals, so
flagging `p_value <= a` reproduces the threshold at rate `a`, and one calibration serves
every operating point. `calibrate(new_normals)` resets the thresholds without refitting.
`save(path)` and `LUNAPipeline.load(path)` keep a fitted pipeline.

Optional input keys: `coarse_logits` (hierarchical heads), `projections`
(contrastive projection head) — scorers that need a missing key are skipped
automatically. `family_analysis(id_data, ood_data)` gives per-family AUROC and
detections at every rate; `report(...)` writes a ranked anomaly table and
`detection_by_far.csv`.

## Objects with several samples

When an object has several samples (spectra, epochs, light-curve windows), put its identifier in
the data dict under `"object_id"`. The calibration, the p-values, the flags and `evaluate` are then
per object: the score of an object is the mean of its samples' ensemble scores (`object_aggregate=
"max"` is also available), the thresholds are set on calibration objects, and the reference and
calibration halves of `fit_blind` / `fit_combiner` keep whole objects together.

```python
val  = {"embeddings": Zva, "logits": Lva, "object_id": obj_va}
test = {"embeddings": Zte, "logits": Lte, "object_id": obj_te}
pipe = LUNAPipeline(target_far=(0.01, 0.05)).fit(train)
pipe.fit_blind(val)                     # per object, because val carries object_id
out = pipe.predict(test)                # out["objects"], out["p_value"], out["flags"][0.01]
```

Calibrate in the unit you report. On ASTRAnet spectra (1,665 validation, 1,668 test and 107 rare
objects, with one to fifty-four spectra each, ten calibration splits), a per-spectrum threshold at
1% flagged 4.2% of the test objects with three or more spectra and 1.0% of those with one.
Calibrating on object scores sets the overall per-object rate, but the rate still depends on the
number of samples: with the maximum over spectra it was 3.0% for objects with three or more spectra
and 0.8% for objects with one, and with the mean 0.03% and 1.4%. The mean also detected more rare
objects than the maximum at the same overall per-object rate, for each of three classifier
backbones (paired gains of 5 to 11 percentage points, positive in 28 of 30 splits), which is why it
is the default. If your objects differ strongly in their number of samples, calibrate separately
within bins of the sample count.

## False-discovery control and prediction sets

`luna.thresholding.conformal_bh(p_value, q)` selects a batch of candidates whose expected fraction
of in-distribution objects is at most `q` (Benjamini-Hochberg on conformal p-values), and
`storey_pi0(p_value)` estimates the in-distribution fraction of the batch. `luna.conformal`
builds split-conformal prediction sets for the known classes, calibrated globally or within strata
(`mondrian_sets`); `joint_strata` defines the four cells of high anomaly score by low confidence,
in which every examined group of ASTRAnet objects reached close to the target coverage.

## Several representations (`luna.multiview`)

`MultiViewDetector` runs LUNA on several representations of the same objects (e.g. two
classifier backbones, or an embedding and physical features) and combines the views into one
statistic (`mean_rank`, `fisher`, `union` or `vote`) that is calibrated at the target rate on
in-distribution data. Do not OR several independently calibrated thresholds; the union of two 1%
rules is a 2% rule. On ASTRAnet, `mean_rank` was the best combination of two backbones (41.8% of
rare objects at 1% per object, against 34.9% for `union`), and a single view on the concatenated
embeddings was better still (43.6%).

## What's inside

```
luna/
  pipeline.py      LUNAPipeline: fit / score / fit_blind / fit_combiner / calibrate /
                   predict / evaluate / family_analysis / save / load (per sample or per object)
  multiview.py     ViewScorer, MultiViewDetector: several representations, one calibrated statistic
  conformal.py     prediction sets: global, Mondrian, joint (anomaly x confidence) strata
  scorers/         16 scorers in 4 families (uncertainty, distance, density, hybrid)
  combiners/       lightgbm (default), logistic, rank-average, xgboost
  thresholding.py  conservative FAR thresholds, conformal p-values, object aggregation,
                   conformal Benjamini-Hochberg, Storey pi0
  viz.py           detection map, complementarity map, ROC/PR, AUROC bars, UMAP
  style.py         publication figure standard (final-size figures, light+dark themes)
  report.py        ranked anomaly reports (CSV / LaTeX)
```

## Evaluation protocol (recommended)

When benchmarking detection on a known set of outliers, avoid leakage: use a
stratified k-fold over the outliers so no object is scored by a combiner that saw
it in training, and calibrate thresholds on a held-out in-distribution split. See
the [ASTRAnet](https://github.com/applecider-ml/ASTRAnet) pipeline for a complete,
worked example of this protocol on astronomical spectra (where LUNA serves as the
anomaly-detection layer of a transient classifier).

## Changes in 0.4

- Per-object calibration: data dicts may carry `object_id`; thresholds, p-values, flags,
  `evaluate` and `family_analysis` are then per object, and the reference/calibration splits keep
  objects together. The default object score is the mean over the object's samples.
- A rate that the calibration set is too small for (n + 1 < 1 / far) gets an infinite threshold
  (`strict=True`, the default since 0.3.1), so that the false-alarm guarantee is never silently
  lost; `strict=False` returns the largest calibration score instead.
- `luna.multiview` (several representations combined into one calibrated statistic) is part of
  the package again, with the conservative order-statistic threshold, several rates, combined
  p-values and per-object mode; the default combination is `mean_rank`.
- `luna.conformal` (global, Mondrian and joint anomaly-by-confidence prediction sets) and
  `conformal_bh` / `storey_pi0` for false-discovery control on conformal p-values.

## Changes in 0.3

- `fit_blind` adds the blind consensus to the pipeline. Its percentile ranks are frozen on
  reference normals, so the score of an object no longer depends on the batch it is scored
  with (in 0.2, `family_analysis` ranked the normals and the outliers together).
- The thresholds are set on in-distribution objects that neither the scorers nor the
  combiner have seen (in 0.2, `fit_combiner` calibrated on its own training normals), with
  the conservative split-conformal order statistic.
- `target_far` accepts several rates; `predict` returns a p-value and flags at each rate,
  and `evaluate` reports the detection rate together with the realised false-alarm rate.

## Demo

`notebooks/demo.ipynb` builds a toy 2-class Gaussian problem, trains a tiny model,
and walks through scoring, family analysis and FAR-thresholded detection.
