# tests/test_head_xgb.py
"""Contract tests for tree-ensemble TIS heads: predict() returns calibrated [0,1] probs."""
import numpy as np
import pytest

from tisiago.head_xgb import fit_lgb_head, fit_rf_head, fit_xgb_head


def _toy(n=400, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    # a learnable signal: positive when first two coords sum high, plus class imbalance
    logit = X[:, 0] + X[:, 1]
    y = (logit + rng.standard_normal(n) * 0.5 > 1.0).astype(int)
    return X, y


@pytest.mark.parametrize("fit", [fit_xgb_head, fit_lgb_head, fit_rf_head])
def test_head_contract(fit):
    Xtr, ytr = _toy(seed=0)
    Xva, yva = _toy(seed=1)
    head = fit(Xtr, ytr, Xva, yva, n_estimators=40, seed=0)
    assert callable(head["predict"])
    assert head["scaler"] is None
    p = head["predict"](Xva)
    assert p.shape == (Xva.shape[0],)
    assert p.min() >= 0.0 and p.max() <= 1.0
    # learns *something*: positives rank above negatives on average
    assert p[yva == 1].mean() > p[yva == 0].mean()


def test_xgb_scale_pos_weight_auto():
    # imbalanced toy: auto scale_pos_weight should be ~ neg/pos, head still calibrates to [0,1]
    Xtr, ytr = _toy(n=800, seed=2)
    Xva, yva = _toy(n=400, seed=3)
    head = fit_xgb_head(Xtr, ytr, Xva, yva, n_estimators=40, scale_pos_weight=None, seed=0)
    p = head["predict"](Xva)
    assert np.isfinite(p).all()
    assert 0.0 <= p.min() and p.max() <= 1.0
