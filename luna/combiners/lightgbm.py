"""LightGBM-based combiner for OOD scores."""
from __future__ import annotations

import numpy as np

from luna.combiners.base import BaseCombiner


class LightGBMCombiner(BaseCombiner):
    """LightGBM classifier for combining OOD scores.

    Args:
        n_estimators: number of boosting rounds (default 100).
        max_depth: maximum tree depth (default 4).
        learning_rate: boosting learning rate (default 0.05).
        random_state: random seed (default 42).
        n_jobs: threads (default 1; the score matrices are small, and LightGBM's default
            of every core can be orders of magnitude slower on a shared node).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LightGBMCombiner:
        try:
            import lightgbm as lgb
        except Exception as e:
            raise ImportError(
                "LightGBM is required for LightGBMCombiner. "
                "Install it with: pip install luna[lightgbm]"
            ) from e
        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def _init_kwargs(self) -> dict:
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
