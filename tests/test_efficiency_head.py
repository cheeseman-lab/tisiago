"""Ridge efficiency head: metrics dict + classifier-thresholding of continuous predictions."""
import numpy as np
import pandas as pd

from tisiago.efficiency_head import _build_target, evaluate_as_classifier, fit_efficiency_head


def test_fit_returns_metrics_and_predict():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8)).astype(np.float32)
    w = rng.standard_normal(8)
    y = X @ w + rng.standard_normal(300) * 0.1
    head = fit_efficiency_head(X[:200], y[:200], X[200:], y[200:], alpha=1.0)
    assert set(["r2_val", "rmse_val", "spearman_val"]).issubset(head["metrics"])
    assert head["metrics"]["r2_val"] > 0.5  # recoverable linear signal
    p = head["predict"](X[:5])
    assert p.shape == (5,)
    expected = head["model"].predict(head["scaler"].transform(X[:5]))
    np.testing.assert_allclose(p, expected, rtol=2e-5, atol=2e-6)


def test_evaluate_as_classifier_keys_and_norm():
    rng = np.random.default_rng(1)
    preds = rng.standard_normal(200) * 5.0  # arbitrary scale, not [0,1]
    y = (preds > preds.mean()).astype(int)
    tx = rng.integers(0, 10, 200)
    out = evaluate_as_classifier(preds, y, tx, budgets=(1.0, 5.0))
    assert "AUPRC" in out and "recall@1.0FP" in out and "recall@5.0FP" in out
    assert 0.0 <= out["AUPRC"] <= 1.0


def test_evaluate_as_classifier_constant_preds_safe():
    # all-equal predictions must not divide by zero
    preds = np.full(50, 3.0)
    y = np.zeros(50, dtype=int)
    y[:5] = 1
    tx = np.arange(50) % 7
    out = evaluate_as_classifier(preds, y, tx, budgets=(1.0,))
    assert np.isfinite(out["AUPRC"])


def test_build_target_excludes_unexpressed_and_aggregates_duplicate_calls():
    m = pd.DataFrame(
        {
            "transcript_id": ["a", "a", "b", "c"],
            "mrna_index": [1, 1, 2, 3],
            "max_norm_HeLa": [np.nan, 3.0, np.nan, np.nan],
            "expressed_HeLa": [False, True, True, False],
        }
    )
    y = _build_target(m, "max_norm_HeLa")
    assert y[0] == y[1] == np.log1p(3.0)
    assert y[2] == 0.0
    assert np.isnan(y[3])
