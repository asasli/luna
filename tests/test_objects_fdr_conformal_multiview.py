"""Object-level calibration, strict thresholds, FDR control, Mondrian sets and multi-view (LUNA v0.4)."""
import numpy as np
import pytest

from luna import LUNAPipeline, MultiViewDetector, ViewScorer
from luna.conformal import coverage, joint_strata, mondrian_sets
from luna.thresholding import aggregate_by_object, calibrate_threshold, conformal_bh, storey_pi0

SCORERS = ["msp", "energy", "mahal_global", "knn"]


def make(n, shift=0.0, seed=0, n_class=3, dim=8):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_class, n)
    centres = np.eye(n_class, dim) * 4
    emb = centres[labels] + rng.normal(size=(n, dim)) + shift
    logits = -((emb[:, None, :] - centres[None]) ** 2).sum(-1) / 2
    return dict(embeddings=emb, logits=logits, labels=labels)


def with_objects(data, n_epochs, seed=0, prefix="o"):
    """Group consecutive samples into objects with 1..n_epochs samples each."""
    rng = np.random.default_rng(seed)
    n, ids, i, k = len(data["labels"]), [], 0, 0
    while i < n:
        m = int(rng.integers(1, n_epochs + 1))
        ids += [f"{prefix}{k}"] * min(m, n - i)
        i += m
        k += 1
    return {**data, "object_id": np.array(ids)}


def test_strict_threshold_is_infinite_when_too_few():
    s = np.arange(1, 50, dtype=float)                     # n = 49 < 99
    assert calibrate_threshold(s, 0.01) == np.inf           # strict by default
    assert calibrate_threshold(s, 0.01, strict=False) == 49.0


def test_aggregate_by_object():
    objs, mx = aggregate_by_object(np.array([1., 5., 2., 3.]), np.array(["b", "b", "a", "a"]), "max")
    assert list(objs) == ["a", "b"] and list(mx) == [3., 5.]
    assert list(aggregate_by_object(np.array([1., 5., 2., 3.]), np.array(["b", "b", "a", "a"]))[1]) == [2.5, 3.]


def test_object_calibration_controls_object_far():
    """A per-sample threshold flags multi-epoch objects too often; per-object calibration does not."""
    train = make(3000, seed=1)
    ref, cal, test = (with_objects(make(n, seed=s), 6, seed=s, prefix=f"s{s}_") for n, s in
                      [(6000, 2), (6000, 3), (30000, 4)])
    per_obj = LUNAPipeline(scorers=SCORERS, target_far=0.05).fit(train)
    per_obj.fit_blind(ref, cal, verbose=False)
    assert per_obj.unit == "object"
    out = per_obj.predict(test)
    assert abs(out["flagged"].mean() - 0.05) < 0.015
    per_spec = LUNAPipeline(scorers=SCORERS, target_far=0.05).fit(train)
    per_spec.fit_blind({k: v for k, v in ref.items() if k != "object_id"},
                       {k: v for k, v in cal.items() if k != "object_id"}, verbose=False)
    spec_flags = per_spec.predict({k: v for k, v in test.items() if k != "object_id"})["flagged"]
    _, obj_flag = aggregate_by_object(spec_flags.astype(float), test["object_id"], "max")
    assert obj_flag.mean() > 0.10                          # roughly the 5% budget times ~3.5 epochs
    with pytest.raises(ValueError):
        per_obj.predict({k: v for k, v in test.items() if k != "object_id"})


def test_object_split_keeps_objects_together():
    data = with_objects(make(500, seed=5), 4, seed=5)
    a, b = LUNAPipeline._split(data, 0.5, 0)
    assert set(a["object_id"]).isdisjoint(b["object_id"])
    assert len(a["object_id"]) + len(b["object_id"]) == 500


def test_evaluate_per_object_with_sample_groups():
    pipe = LUNAPipeline(scorers=SCORERS, target_far=(0.01, 0.05)).fit(make(3000, seed=1))
    pipe.fit_blind(with_objects(make(6000, seed=2), 3, seed=2), verbose=False)
    odd = with_objects(make(300, shift=3.0, seed=8), 3, seed=8, prefix="x")
    fam = np.where(np.arange(300) < 150, "a", "b")
    tab = pipe.evaluate(with_objects(make(3000, seed=7), 3, seed=7, prefix="t"), odd, groups=fam)
    assert set(tab.unit) == {"object"} and {"a", "b", "all"} <= set(tab.group)


def test_conformal_bh_controls_fdr():
    rng = np.random.default_rng(0)
    fdr = []
    for r in range(200):
        cal = rng.normal(size=500)
        new = np.r_[rng.normal(size=900), rng.normal(3.5, 1, size=100)]
        is_out = np.r_[np.zeros(900, bool), np.ones(100, bool)]
        from luna.thresholding import p_values
        f = conformal_bh(p_values(new, cal), q=0.1)
        fdr.append((f & ~is_out).sum() / max(f.sum(), 1))
    assert np.mean(fdr) <= 0.12
    p = p_values(rng.normal(size=5000), rng.normal(size=999))
    assert abs(storey_pi0(p) - 1.0) < 0.06


def test_mondrian_sets_cover_each_stratum():
    rng = np.random.default_rng(1)
    n, C = 6000, 4
    y = rng.integers(0, C, n)
    hard = rng.random(n) < 0.3                             # a stratum where the classifier is worse
    logit = rng.normal(size=(n, C)) + np.where(hard, 0.5, 3.0)[:, None] * np.eye(C)[y]
    p = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
    cal, new = np.arange(n) < n // 2, np.arange(n) >= n // 2
    glob = mondrian_sets(p[cal], y[cal], None, p[new], None, 0.1)
    mond = mondrian_sets(p[cal], y[cal], hard[cal].astype(int), p[new], hard[new].astype(int), 0.1)
    h = hard[new]
    assert coverage(glob, y[new], h)["coverage"] < 0.85
    assert abs(coverage(mond, y[new], h)["coverage"] - 0.9) < 0.03
    assert abs(coverage(mond, y[new], ~h)["coverage"] - 0.9) < 0.03
    assert set(np.unique(joint_strata(np.r_[0, 1, 0, 1], np.r_[.9, .9, .1, .1], .5, .5))) == {0, 1, 2, 3}


@pytest.mark.parametrize("combination", ["mean_rank", "fisher", "union", "vote"])
def test_multiview_calibrated_per_object(combination):
    rng = np.random.default_rng(3)
    def views(n, seed, shift=0.0, prefix="o"):
        a = with_objects(make(n, shift=shift, seed=seed), 4, seed=seed, prefix=prefix)
        b = {**make(n, shift=shift, seed=seed + 100), "object_id": a["object_id"]}
        return {"v1": a, "v2": b}
    det = MultiViewDetector({"v1": ViewScorer(SCORERS), "v2": ViewScorer(SCORERS)},
                            combination=combination, k=2, target_far=(0.05, 0.10))
    tr = {"v1": make(3000, seed=1), "v2": make(3000, seed=101)}
    det.fit(tr, views(4000, 2, prefix="r"), views(8000, 3, prefix="c"))
    out = det.predict(views(20000, 4, prefix="t"))
    assert out["objects"] is not None
    assert abs(out["flags"][0.05].mean() - 0.05) < 0.02
    assert np.array_equal(out["p_value"] <= 0.05, out["flags"][0.05])
    odd = det.predict(views(400, 5, shift=3.0, prefix="x"))
    assert odd["flagged"].mean() > 0.5
