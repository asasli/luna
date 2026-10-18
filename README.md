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

## What's inside

```
luna/
  pipeline.py      LUNAPipeline: fit / score / fit_blind / fit_combiner / calibrate /
                   predict / evaluate / family_analysis / save / load
  scorers/         16 scorers in 4 families (uncertainty, distance, density, hybrid)
  combiners/       lightgbm (default), logistic, rank-average, xgboost
  thresholding.py  conservative FAR thresholds and conformal p-values
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
