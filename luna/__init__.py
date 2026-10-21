"""LUNA — Library for Uncertainty and Novelty Analysis.

Classifier-agnostic out-of-distribution / novelty detection: 16 anomaly scores
in four families (uncertainty, distance, density, hybrid) over any model's
embeddings + logits, combined by a trainable ensemble (LightGBM by default) or
blindly by mean percentile rank, calibrated per sample or per object at a target
false-alarm rate. Optional modules: `luna.multiview` (several representations
combined into one calibrated statistic) and `luna.conformal` (prediction sets
stratified by novelty and confidence).
"""
__version__ = "0.4.1"

from luna.pipeline import LUNAPipeline, OBJECT_KEY
from luna.multiview import MultiViewDetector, ViewScorer

__all__ = ["LUNAPipeline", "OBJECT_KEY", "MultiViewDetector", "ViewScorer"]
