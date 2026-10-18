"""Calibration, p-values, multi-rate flags and batch independence (LUNA v0.3)."""
import numpy as np
import pytest

from luna import LUNAPipeline
from luna.thresholding import calibrate_threshold, p_values

SCORERS = ["msp", "energy", "mahal_global", "knn"]


def make(n, shift=0.0, seed=0, n_class=3, dim=8):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_class, n)
    centres = np.eye(n_class, dim) * 4
    emb = centres[labels] + rng.normal(size=(n, dim)) + shift
    logits = -((emb[:, None, :] - centres[None]) ** 2).sum(-1) / 2
    return dict(embeddings=emb, logits=logits, labels=labels)


@pytest.fixture(scope="module")
def blind():
    pipe = LUNAPipeline(scorers=SCORERS, target_far=(0.01, 0.05, 0.10)).fit(make(3000, seed=1))
    return pipe.fit_blind(make(4000, seed=2), verbose=False)


def test_threshold_is_conservative_order_statistic():
    s = np.arange(1, 100, dtype=float)          # n = 99, (n + 1)(1 - 0.05) = 95
    assert calibrate_threshold(s, 0.05) == 95.0
    assert calibrate_threshold(s, 0.001) == 99.0  # too few objects: the maximum


def test_p_values_match_thresholds():
    rng = np.random.default_rng(0)
    cal, new = rng.normal(size=999), rng.normal(size=5000)
    p = p_values(new, cal)
    for a in (0.01, 0.05, 0.10):
        assert np.array_equal(p <= a, new > calibrate_threshold(cal, a))


def test_realised_fpr_close_to_target(blind):
    out = blind.predict(make(20000, seed=3))
    for a, f in out["flags"].items():
        assert abs(f.mean() - a) < 0.35 * a + 0.004, (a, f.mean())
    assert out["flagged"] is out["flags"][0.01]
    assert set(out["thresholds"]) == {0.01, 0.05, 0.10}


def test_p_values_uniform_on_normals(blind):
    p = blind.predict(make(20000, seed=4))["p_value"]
    assert abs(np.mean(p <= 0.2) - 0.2) < 0.03


def test_blind_score_independent_of_batch(blind):
    normals, odd = make(500, seed=5), make(50, shift=3.0, seed=6)
    alone = blind.predict(odd)["ensemble_score"]
    both = {k: np.concatenate([odd[k], normals[k]]) for k in odd}
    together = blind.predict(both)["ensemble_score"][:50]
    assert np.array_equal(alone, together)


def test_evaluate_table(blind):
    tab = blind.evaluate(make(3000, seed=7), make(200, shift=3.0, seed=8),
                         groups=np.repeat(["a", "b"], 100), methods=True)
    ens = tab[(tab.rule == "ensemble") & (tab.group == "all")].set_index("target_far")
    assert list(ens.index) == [0.01, 0.05, 0.10]
    assert ens.det_rate.is_monotonic_increasing
    assert set(tab.rule) == {"ensemble", *SCORERS}
    assert (ens.auroc > 0.9).all()


def test_fit_combiner_calibrates_on_held_out_half():
    pipe = LUNAPipeline(scorers=SCORERS, combiner="logistic", target_far=(0.01, 0.05))
    pipe.fit(make(3000, seed=1))
    pipe.fit_combiner(make(4000, seed=2), make(300, shift=3.0, seed=9), verbose=False)
    assert len(pipe._cal_scores) == 2000
    out = pipe.predict(make(20000, seed=10))
    assert abs(out["flags"][0.05].mean() - 0.05) < 0.02


def test_family_analysis_has_every_rate(blind):
    fa = blind.family_analysis(make(2000, seed=11), make(100, shift=3.0, seed=12))
    for info in fa.values():
        assert set(info["det_rate_by_far"]) == {0.01, 0.05, 0.10}


def test_save_load_roundtrip(blind, tmp_path):
    blind.save(tmp_path / "p.pkl")
    again = LUNAPipeline.load(tmp_path / "p.pkl")
    x = make(100, seed=13)
    assert np.array_equal(again.predict(x)["p_value"], blind.predict(x)["p_value"])


def test_scalar_target_far_still_works():
    pipe = LUNAPipeline(scorers=SCORERS, target_far=0.05).fit(make(2000, seed=1))
    pipe.fit_blind(make(2000, seed=2), verbose=False)
    assert pipe.target_far == 0.05 and pipe.fars == (0.05,)
