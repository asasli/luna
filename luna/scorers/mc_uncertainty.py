"""Monte Carlo Dropout uncertainty scorers."""

from __future__ import annotations

import numpy as np

from luna.scorers.base import BaseScorer, register_scorer


_EPS = 1e-10


def _get_probs_t(data):
    """
    Return MC probabilities in canonical shape (T, N, K).

    Accepted:
        (T, N, K)
        (N, T, K)
    """
    if "probs_t" not in data:
        raise KeyError(
            "MC uncertainty scorer requires 'probs_t'."
        )

    p = np.asarray(data["probs_t"], dtype=np.float64)

    if p.ndim != 3:
        raise ValueError(
            f"'probs_t' must be 3-D, got shape {p.shape}"
        )

    if "logits" in data:
        n = len(data["logits"])
    elif "embeddings" in data:
        n = len(data["embeddings"])
    else:
        raise KeyError(
            "Need logits or embeddings to infer sample dimension."
        )

    if p.shape[1] == n:
        return p

    if p.shape[0] == n:
        return np.transpose(p, (1, 0, 2))

    raise ValueError(
        f"Cannot align probs_t shape {p.shape} with N={n}"
    )


def _mean_probs(p):
    return p.mean(axis=0)


def _entropy(p):
    p = np.clip(p, _EPS, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


class _MCStatelessScorer(BaseScorer):
    family = "mc_uncertainty"
    requires = {"probs_t", "logits"}

    def fit(self, train_data):
        self._fitted = True
        return self


@register_scorer
class MCPredictiveEntropyScorer(_MCStatelessScorer):
    name = "mc_predictive_entropy"

    def score(self, data):
        p = _get_probs_t(data)
        return _entropy(_mean_probs(p))


@register_scorer
class MCMutualInformationScorer(_MCStatelessScorer):
    name = "mc_mutual_information"

    def score(self, data):
        p = _get_probs_t(data)
        mean_p = _mean_probs(p)
        predictive_entropy = _entropy(mean_p)
        expected_entropy = _entropy(p).mean(axis=0)
        return predictive_entropy - expected_entropy


@register_scorer
class MCPredictedClassStdScorer(_MCStatelessScorer):
    name = "mc_predicted_class_std"

    def score(self, data):
        p = _get_probs_t(data)
        mean_p = _mean_probs(p)
        predicted_class = mean_p.argmax(axis=1)
        n = mean_p.shape[0]

        predicted_class_probs = p[
            :,
            np.arange(n),
            predicted_class,
        ]

        return predicted_class_probs.std(
            axis=0,
            ddof=1 if p.shape[0] > 1 else 0,
        )


@register_scorer
class MCMeanProbabilityStdScorer(_MCStatelessScorer):
    name = "mc_mean_probability_std"

    def score(self, data):
        p = _get_probs_t(data)

        probability_std = p.std(
            axis=0,
            ddof=1 if p.shape[0] > 1 else 0,
        )

        return probability_std.mean(axis=1)


@register_scorer
class MCMaxProbabilityStdScorer(_MCStatelessScorer):
    name = "mc_max_probability_std"

    def score(self, data):
        p = _get_probs_t(data)

        probability_std = p.std(
            axis=0,
            ddof=1 if p.shape[0] > 1 else 0,
        )

        return probability_std.max(axis=1)
