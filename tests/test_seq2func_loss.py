"""Tests for the masked Poisson-multinomial loss."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from tisiago.seq2func.loss import codon_phase_multinomial, poisson_multinomial  # noqa: E402


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


def _periodic(length=60, channels=2, phase=0):
    y = np.zeros((1, length, channels), np.float32)
    y[0, phase::3, :] = 4.0
    y[0, (phase + 1) % 3 :: 3, :] = 1.0
    return jnp.asarray(y), jnp.ones(y.shape, bool)


@pytest.mark.parametrize("block", [1, 2, 10])
def test_codon_phase_zero_at_matching_phase_composition(block):
    y, mask = _periodic()
    assert float(codon_phase_multinomial(2.0 * y, y, mask, (0, 1), block)) == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.parametrize("block", [1, 2, 10])
def test_codon_phase_positive_for_shifted_frame(block):
    y, mask = _periodic()
    shifted, _ = _periodic(phase=1)
    assert float(codon_phase_multinomial(shifted, y, mask, (0, 1), block)) > 0.1


def test_codon_phase_block_one_is_per_triplet():
    rng = np.random.default_rng(1)
    y = jnp.asarray(rng.poisson(1.0, size=(2, 31, 1)).astype(np.float32))
    pred = jnp.asarray(rng.uniform(0.1, 2.0, size=(2, 31, 1)).astype(np.float32))
    mask = jnp.ones(y.shape, bool)
    t = np.asarray(y)[:, :30, 0].reshape(2, 10, 3)
    p = np.asarray(pred)[:, :30, 0].reshape(2, 10, 3)
    n = t.sum(-1)
    qt = t / np.maximum(n[..., None], 1e-7)
    qp = p / p.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(qt > 0, qt * np.log(qt / qp), 0.0).sum(-1)
    expected = (n * kl).sum() / n.sum()
    assert float(codon_phase_multinomial(pred, y, mask, (0,), 1)) == pytest.approx(
        expected, rel=1e-4
    )


def test_codon_phase_block_pools_phase_counts():
    # Within each triplet the phase is off, but pooled over 2 triplets the composition matches.
    y = jnp.asarray(np.array([1, 0, 0, 0, 1, 0], np.float32)[None, :, None])
    pred = jnp.asarray(np.array([0, 1, 0, 1, 0, 0], np.float32)[None, :, None] + 1e-6)
    mask = jnp.ones(y.shape, bool)
    assert float(codon_phase_multinomial(pred, y, mask, (0,), 1)) > 1.0
    assert float(codon_phase_multinomial(pred, y, mask, (0,), 2)) == pytest.approx(0.0, abs=1e-4)


def test_codon_phase_skips_masked_blocks():
    y, mask = _periodic()
    shifted, _ = _periodic(phase=1)
    pred = y.at[:, :30, :].set(shifted[:, :30, :])  # wrong only in the first block of 10 codons
    mask = mask.at[:, 7, :].set(False)  # one masked position kills the whole block
    assert float(codon_phase_multinomial(pred, y, mask, (0, 1), 10)) == pytest.approx(0.0, abs=1e-6)


def test_codon_phase_ignores_other_channels():
    y, mask = _periodic()
    shifted, _ = _periodic(phase=1)
    pred = y.at[:, :, 1].set(shifted[:, :, 1])
    assert float(codon_phase_multinomial(pred, y, mask, (0,), 10)) == pytest.approx(0.0, abs=1e-6)
    assert float(codon_phase_multinomial(pred, y, mask, (1,), 10)) > 0.1


def test_codon_phase_zero_without_counts_or_channels():
    y, mask = _periodic()
    assert float(codon_phase_multinomial(y, jnp.zeros_like(y), mask, (0, 1), 10)) == 0.0
    assert float(codon_phase_multinomial(y, y, mask, (), 10)) == 0.0


def test_codon_phase_gradient_is_finite():
    y, mask = _periodic()
    pred = jnp.ones_like(y).at[:, :10, :].set(0.0)
    grad = jax.grad(lambda p: codon_phase_multinomial(p, y, mask, (0, 1), 10))(pred)
    assert bool(jnp.all(jnp.isfinite(grad))) and float(jnp.abs(grad).sum()) > 0


def test_frame_weight_zero_leaves_loss_unchanged():
    y, mask = _arrays()
    pred = jnp.ones_like(y) * 1.7
    base = poisson_multinomial(pred, y, mask)
    out = poisson_multinomial(pred, y, mask, frame_weight=0.0, frame_channels=(0, 1))
    assert np.asarray(out["loss"]).tobytes() == np.asarray(base["loss"]).tobytes()
    assert float(out["frame"]) > 0
    assert float(base["frame"]) == 0.0


def test_frame_weight_adds_weighted_frame_term():
    y, mask = _arrays()
    pred = jnp.ones_like(y) * 1.7
    base = poisson_multinomial(pred, y, mask)
    out = poisson_multinomial(pred, y, mask, frame_weight=2.5, frame_channels=(0,), frame_block=3)
    frame = codon_phase_multinomial(pred, y, mask, (0,), 3)
    assert float(out["frame"]) == pytest.approx(float(frame))
    assert float(out["loss"]) == pytest.approx(float(base["loss"]) + 2.5 * float(frame), rel=1e-6)
