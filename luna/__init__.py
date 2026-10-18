"""LUNA — Library for Uncertainty and Novelty Analysis.

Classifier-agnostic out-of-distribution / novelty detection: 16 anomaly scores
in four families (uncertainty, distance, density, hybrid) over any model's
embeddings + logits, combined by a trainable ensemble (LightGBM by default).
"""
__version__ = "0.3.0"

from luna.pipeline import LUNAPipeline

__all__ = ["LUNAPipeline"]
