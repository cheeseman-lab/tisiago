import warnings

import numpy as np
import pandas as pd

from tisiago.tracks.metrics import gene_totals, log_pearson, profile_kl


def _yt():
    rng = np.random.default_rng(0)
    y = rng.poisson(3.0, size=(50, 2)).astype(np.float32) + 1
    return y, np.ones_like(y, dtype=bool)


def test_profile_kl_zero_on_identity():
    y, m = _yt()
    np.testing.assert_allclose(profile_kl(y, y, m), 0.0, atol=1e-5)


def test_profile_kl_scale_free():
    y, m = _yt()
    np.testing.assert_allclose(profile_kl(2 * y, y, m), 0.0, atol=1e-5)


def test_profile_kl_positive_when_different():
    y, m = _yt()
    assert (profile_kl(y[::-1].copy(), y, m) > 0).all()


def test_profile_kl_fully_masked_channel_nan():
    y, m = _yt()
    m[:, 1] = False
    out = profile_kl(y, y, m)
    assert out.shape == (2,)
    assert np.isnan(out[1]) and not np.isnan(out[0])


def test_profile_kl_ignores_masked_positions():
    y, m = _yt()
    m[:10] = False
    pred = y.copy()
    pred[:10] = 1000.0
    np.testing.assert_allclose(profile_kl(pred, y, m), 0.0, atol=1e-5)


def test_gene_totals_basic_and_order():
    track = np.arange(20, dtype=np.float32).reshape(10, 2)
    genes = pd.DataFrame({"start": [105, 100], "end": [108, 102]})
    out = gene_totals(track, genes, 100)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], track[5:8].sum(0))
    np.testing.assert_allclose(out[1], track[0:2].sum(0))


def test_gene_totals_clips_both_edges():
    track = np.ones((10, 2), dtype=np.float32)
    genes = pd.DataFrame({"start": [95, 108, 90], "end": [103, 120, 130]})
    out = gene_totals(track, genes, 100)
    np.testing.assert_allclose(out[:, 0], [3, 2, 10])


def test_gene_totals_outside_window_zero():
    track = np.ones((10, 1), dtype=np.float32)
    genes = pd.DataFrame({"start": [0, 200], "end": [50, 300]})
    np.testing.assert_allclose(gene_totals(track, genes, 100), 0.0)


def test_log_pearson_proportional_is_one():
    obs = np.array([[1.0], [10.0], [100.0], [1000.0]])
    np.testing.assert_allclose(log_pearson(obs * 3, obs), 1.0, atol=1e-2)
    x = np.log1p(np.arange(5.0))[:, None]
    assert log_pearson(np.expm1(x), np.expm1(2 * x))[0] > 0.999


def test_log_pearson_too_few_rows_nan():
    a = np.array([[1.0], [2.0]])
    assert np.isnan(log_pearson(a, a)).all()


def test_log_pearson_zero_variance_nan_no_warning():
    a = np.ones((5, 1))
    b = np.arange(5.0)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(log_pearson(a, b)).all()


def test_profile_kl_matches_dense_formula_on_sparse_targets():
    rng = np.random.default_rng(1)
    target = rng.poisson(0.05, size=(500, 4)).astype(np.float32)
    pred = rng.gamma(1.0, size=(500, 4)).astype(np.float32)
    pred[::7, 1] = 0.0
    mask = rng.random((500, 4)) > 0.2
    p = np.where(mask, pred.astype(np.float64), 0.0)
    t = np.where(mask, target.astype(np.float64), 0.0)
    pt, pp = t / t.sum(axis=0), p / p.sum(axis=0)
    safe = np.where(pt > 0, pt, 1.0)
    expected = np.where(
        pt > 0, pt * (np.log(safe) - np.log(np.maximum(np.where(pt > 0, pp, 1.0), 1e-7))), 0.0
    ).sum(axis=0)
    np.testing.assert_allclose(profile_kl(pred, target, mask), expected, rtol=1e-10)
