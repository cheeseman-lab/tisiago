"""Tests for the masked Poisson-multinomial loss."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from tisiago.tracks.loss import codon_phase_multinomial, poisson_multinomial  # noqa: E402


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


# --- local (per-block) multinomial on selected channels ---


def _local_reference(pred, target, mask, channels, block):
    pred = np.where(mask, np.asarray(pred, np.float64), 0.0)[..., list(channels)]
    target = np.where(mask, np.maximum(np.asarray(target, np.float64), 0), 0.0)[..., list(channels)]
    n = (pred.shape[1] // block) * block
    num = den = 0.0
    for b in range(pred.shape[0]):
        for lo in range(0, n, block):
            for c in range(len(channels)):
                t, p = target[b, lo : lo + block, c], pred[b, lo : lo + block, c]
                if t.sum() <= 0:
                    continue
                qt, qp = t / t.sum(), p / max(p.sum(), 1e-7)
                kl = sum(a * (np.log(a) - np.log(max(q, 1e-7))) for a, q in zip(qt, qp) if a > 0)
                num, den = num + t.sum() * kl, den + t.sum()
    return num / den if den else 0.0


def test_local_multinomial_matches_reference_and_ignores_scale():
    from tisiago.tracks.loss import local_multinomial

    rng = np.random.default_rng(3)
    target = rng.poisson(1.0, (2, 70, 4)).astype(np.float32)
    pred = rng.gamma(1.0, 1.0, (2, 70, 4)).astype(np.float32)
    mask = np.ones(target.shape, bool)
    mask[0, 5:20, 1] = False
    got = float(local_multinomial(jnp.asarray(pred), jnp.asarray(target), mask, (0, 1), 16))
    assert got == pytest.approx(_local_reference(pred, target, mask, (0, 1), 16), rel=1e-4)
    # per-block shape only: scaling each block of the prediction leaves it unchanged
    scaled = pred.copy()
    scaled[:, :16] *= 7.0
    again = float(local_multinomial(jnp.asarray(scaled), jnp.asarray(target), mask, (0, 1), 16))
    assert again == pytest.approx(got, rel=1e-4)
    assert float(local_multinomial(jnp.asarray(target), jnp.asarray(target), mask, (0, 1), 16)) == (
        pytest.approx(0.0, abs=1e-6)
    )
    # no channels, or a window shorter than one block: 0
    assert float(local_multinomial(jnp.asarray(pred), jnp.asarray(target), mask, (), 16)) == 0.0
    assert float(local_multinomial(jnp.asarray(pred), jnp.asarray(target), mask, (0,), 128)) == 0.0


def test_poisson_multinomial_local_term_enters_only_with_weight():
    y, mask = _arrays()
    pred = jnp.ones_like(y)
    base = poisson_multinomial(pred, y, mask)
    off = poisson_multinomial(pred, y, mask, local_weight=0.0, local_channels=(0, 1))
    on = poisson_multinomial(pred, y, mask, local_weight=3.0, local_channels=(0, 1), local_block=8)
    assert float(off["loss"]) == pytest.approx(float(base["loss"]))
    assert float(on["local"]) > 0
    expected = float(base["loss"]) + 3.0 * float(on["local"])
    assert float(on["loss"]) == pytest.approx(expected, rel=1e-5)
    assert "local" in base and float(base["local"]) == 0.0
    grad = jax.grad(
        lambda p: poisson_multinomial(p, y, mask, local_weight=3.0, local_channels=(0,))["loss"]
    )(pred)
    assert bool(jnp.isfinite(grad).all())


def test_local2_off_is_bit_identical_and_block32_hand_example():
    y, mask = _arrays()
    pred = jnp.ones_like(y) * 1.5
    base = poisson_multinomial(pred, y, mask, local_weight=2.0, local_channels=(0,))
    off = poisson_multinomial(
        pred, y, mask, local_weight=2.0, local_channels=(0,), local2_channels=(1,)
    )
    assert float(base["loss"]) == float(off["loss"]) and float(off["local2"]) == 0.0
    on = poisson_multinomial(
        pred,
        y,
        mask,
        local_weight=2.0,
        local_channels=(0,),
        local2_weight=7.0,
        local2_channels=(1,),
        local2_block=8,
    )
    from tisiago.tracks.loss import local_multinomial

    ref = float(local_multinomial(pred, y, mask, (1,), 8))
    assert float(on["local2"]) == pytest.approx(ref)
    assert float(on["loss"]) == pytest.approx(float(base["loss"]) + 7.0 * ref, rel=1e-5)
    # one 32-position block, uniform prediction, all counts at one position: KL = log 32
    t = jnp.zeros((1, 32, 1)).at[0, 5, 0].set(4.0)
    m = jnp.ones((1, 32, 1), bool)
    got = poisson_multinomial(
        jnp.ones((1, 32, 1)), t, m, local2_weight=1.0, local2_channels=(0,), local2_block=32
    )["local2"]
    assert float(got) == pytest.approx(np.log(32.0), rel=1e-5)
