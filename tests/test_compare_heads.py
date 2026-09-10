"""compare_heads recomputes the headline metrics per head from saved .npz prediction files."""
import numpy as np

from scripts.compare_heads import metrics_for


def _write_npz(path, with_noncog=True, seed=0):
    rng = np.random.default_rng(seed)
    n = 2000
    klass_cog = rng.random(n) < 0.7
    y = (klass_cog & (rng.random(n) < 0.1)).astype(int)
    tx = rng.integers(0, 50, n)
    # a head that ranks positives a bit higher
    p = rng.random(n) * 0.3 + y * 0.4
    d = {"y": y, "cognate": klass_cog, "tx": tx.astype(str), "p::HeadA": p}
    if with_noncog:
        d["noncog"] = ~klass_cog
    np.savez(path, **d)


def test_metrics_for_classifier_npz(tmp_path):
    f = tmp_path / "preds_a.npz"
    _write_npz(f, with_noncog=True)
    rows = metrics_for(f)
    assert len(rows) == 1
    r = rows[0]
    assert r["head"] == "HeadA"
    assert 0.0 <= r["AUPRC"] <= 1.0
    assert "oracle_recall@1FP" in r and "noncog_mean_p_at_oracle" in r
    assert 0.0 <= r["oracle_recall@1FP"] <= 1.0


def test_metrics_for_handles_missing_noncog(tmp_path):
    f = tmp_path / "preds_reg.npz"
    _write_npz(f, with_noncog=False)
    rows = metrics_for(f)
    assert len(rows) == 1
    # grounding unavailable -> NaN, not a crash
    assert np.isnan(rows[0]["noncog_mean_p_at_oracle"])
