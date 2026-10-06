"""Tests for the masked Poisson-multinomial loss."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from tisiago.seq2func.loss import poisson_multinomial  # noqa: E402


def _arrays():
    rng = np.random.default_rng(0)
    y = rng.poisson(2.0, size=(2, 64, 3)).astype(np.float32)
    y[:, :, 2] = 0  # an all-zero channel
    return jnp.asarray(y), jnp.ones(y.shape, bool)


def test_zero_at_perfect_prediction():
    y, mask = _arrays()
    out = poisson_multinomial(y, y, mask)
    assert float(out["loss"]) == pytest.approx(0.0, abs=1e-5)


def test_positive_and_finite_when_wrong():
    y, mask = _arrays()
    out = poisson_multinomial(jnp.ones_like(y), y, mask)
    assert np.isfinite(float(out["loss"])) and float(out["loss"]) > 0


def test_masked_positions_are_ignored():
    y, mask = _arrays()
    pred = y.at[:, :8, :].set(100.0)  # wrong only where masked
    mask = mask.at[:, :8, :].set(False)
    assert float(poisson_multinomial(pred, y, mask)["loss"]) == pytest.approx(0.0, abs=1e-5)


def test_gradient_is_finite_with_zero_channel_and_zero_predictions():
    y, mask = _arrays()
    pred = jnp.ones_like(y).at[:, :10, :].set(0.0)
    grad = jax.grad(lambda p: poisson_multinomial(p, y, mask)["loss"])(pred)
    assert bool(jnp.all(jnp.isfinite(grad)))
