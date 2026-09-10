"""train_heads_dense routes to the selected classifier; logistic path is unchanged."""
import numpy as np
import pandas as pd
import pytest

import tisiago.dense_caller as dc


@pytest.fixture
def tiny_store(tmp_path):
    """A minimal scan store: 2 keys, train+val+test rows, cognate+non_cognate classes."""
    n = 600
    rng = np.random.default_rng(0)
    # two tiny keys under embeddings/a/ so keys=["a/x.npy","a/z.npy"] resolve
    (tmp_path / "embeddings" / "a").mkdir(parents=True)
    np.save(tmp_path / "embeddings" / "a" / "x.npy", rng.standard_normal((n, 4)).astype(np.float16))
    np.save(tmp_path / "embeddings" / "a" / "z.npy", rng.standard_normal((n, 3)).astype(np.float16))
    split = np.array(["train"] * 300 + ["val"] * 150 + ["test"] * 150)
    klass = rng.choice(["AUG", "near_cognate", "non_cognate"], n, p=[0.1, 0.6, 0.3])
    y = ((klass != "non_cognate") & (rng.random(n) < 0.2)).astype(int)
    m = pd.DataFrame({
        "row_idx": np.arange(n), "transcript_id": rng.integers(0, 20, n),
        "codon_class": klass, "split": split, "label_tis": y,
    })
    m.to_parquet(tmp_path / "manifest.parquet")
    return tmp_path


def test_logistic_unchanged(tiny_store):
    keys = ["a/x.npy", "a/z.npy"]
    heads = dc.train_heads_dense(tiny_store, keys=keys, neg_cap=1000, head="logistic")
    assert set(heads) == {"Dense(bal)", "Dense(None)"}


def test_xgb_route(tiny_store):
    keys = ["a/x.npy", "a/z.npy"]
    heads = dc.train_heads_dense(tiny_store, keys=keys, neg_cap=1000, head="xgboost",
                                 tree_params={"n_estimators": 20})
    assert set(heads) == {"Dense(xgb)"}
    p = heads["Dense(xgb)"]["predict"](np.zeros((5, 7), dtype=np.float32))
    assert p.shape == (5,) and p.min() >= 0.0 and p.max() <= 1.0


def test_unknown_head_raises(tiny_store):
    with pytest.raises((ValueError, KeyError)):
        dc.train_heads_dense(tiny_store, keys=["a/x.npy", "a/z.npy"], head="banana")


def test_chunked_linear_predictions_match_direct_head(tiny_store):
    keys = ["a/x.npy", "a/z.npy"]
    X = dc._load_full(tiny_store / "embeddings", keys)
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, len(X))
    head = dc.fit_calibrated_head(X[:300], y[:300], X[300:450], y[300:450])
    rows = np.arange(450, 600)
    expected = head["predict"](X[rows])
    actual = dc._predict_chunked(
        tiny_store / "embeddings", keys, {"linear": head}, rows, chunk=37
    )["linear"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
