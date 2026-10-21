import numpy as np

from luna.scorers import SCORER_REGISTRY


def _reference_scores(probs_t):
    eps = 1e-10

    mean_probs = probs_t.mean(axis=0)

    clipped_mean = np.clip(mean_probs, eps, 1.0)
    predictive_entropy = -(
        clipped_mean * np.log(clipped_mean)
    ).sum(axis=-1)

    clipped_passes = np.clip(probs_t, eps, 1.0)
    pass_entropy = -(
        clipped_passes * np.log(clipped_passes)
    ).sum(axis=-1)

    mutual_information = (
        predictive_entropy - pass_entropy.mean(axis=0)
    )

    probability_std = probs_t.std(axis=0, ddof=1)

    pred_class = mean_probs.argmax(axis=1)
    n = mean_probs.shape[0]

    pred_class_probs = probs_t[
        :,
        np.arange(n),
        pred_class,
    ]

    return {
        "mc_predictive_entropy":
            predictive_entropy,
        "mc_mutual_information":
            mutual_information,
        "mc_predicted_class_std":
            pred_class_probs.std(axis=0, ddof=1),
        "mc_mean_probability_std":
            probability_std.mean(axis=1),
        "mc_max_probability_std":
            probability_std.max(axis=1),
    }


def test_mc_scorers_match_reference():
    probs_t = np.array(
        [
            [[0.70, 0.20, 0.10],
             [0.20, 0.60, 0.20]],

            [[0.60, 0.30, 0.10],
             [0.30, 0.50, 0.20]],

            [[0.80, 0.10, 0.10],
             [0.10, 0.70, 0.20]],
        ],
        dtype=float,
    )

    data = {
        "probs_t": probs_t,
        "logits": np.zeros((2, 3)),
    }

    expected = _reference_scores(probs_t)

    for name, expected_score in expected.items():
        scorer = SCORER_REGISTRY[name]()
        scorer.fit({})
        actual = scorer.score(data)

        np.testing.assert_allclose(
            actual,
            expected_score,
            rtol=1e-10,
            atol=1e-12,
        )


def test_mc_scorers_accept_ntk_layout():
    rng = np.random.default_rng(42)

    raw = rng.random((5, 4, 3))
    probs_t = raw / raw.sum(axis=-1, keepdims=True)

    data_tnk = {
        "probs_t": probs_t,
        "logits": np.zeros((4, 3)),
    }

    data_ntk = {
        "probs_t": np.transpose(probs_t, (1, 0, 2)),
        "logits": np.zeros((4, 3)),
    }

    names = [
        "mc_predictive_entropy",
        "mc_mutual_information",
        "mc_predicted_class_std",
        "mc_mean_probability_std",
        "mc_max_probability_std",
    ]

    for name in names:
        scorer = SCORER_REGISTRY[name]()
        scorer.fit({})

        a = scorer.score(data_tnk)
        b = scorer.score(data_ntk)

        np.testing.assert_allclose(a, b)
