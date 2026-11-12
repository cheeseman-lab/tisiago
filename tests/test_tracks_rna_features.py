"""RNA-input features computed in JAX inside the rnacond head."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tisiago.tracks.augment import reverse_complement, shift  # noqa: E402
from tisiago.tracks.rna_features import (  # noqa: E402
    NUM_RNA_FEATURES,
    RNA_FEATURE_NAMES,
    RNA_WIDTHS,
    box_mean,
    rna_features,
)


def _box_reference(x: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        out[i] = x[max(0, i - half) : i + half + 1].sum(axis=0) / width
    return out


def test_feature_names_and_widths():
    assert RNA_WIDTHS == (9, 33, 129, 513)
    assert NUM_RNA_FEATURES == 10 == len(RNA_FEATURE_NAMES)
    assert RNA_FEATURE_NAMES[:5] == (
        "plus_log1p",
        "plus_box9",
        "plus_box33",
        "plus_box129",
        "plus_box513",
    )
    assert RNA_FEATURE_NAMES[5] == "minus_log1p"


def test_box_mean_on_hand_array():
    x = jnp.asarray([0, 0, 9, 0, 0, 0, 3], jnp.float32)[None, :, None]
    out = np.asarray(box_mean(x, 3))[0, :, 0]
    np.testing.assert_allclose(out, [0, 3, 3, 3, 0, 1, 1], atol=1e-6)
    # zero padding past the ends: divide by the full width
    edge = np.asarray(box_mean(jnp.ones((1, 4, 1)), 3))[0, :, 0]
    np.testing.assert_allclose(edge, [2 / 3, 1, 1, 2 / 3], atol=1e-6)


@pytest.mark.parametrize("width", [1, 9, 33, 129, 513])
def test_box_mean_matches_reference_and_keeps_length(width):
    rng = np.random.default_rng(width)
    x = rng.poisson(0.5, (300, 2)).astype(np.float32)
    out = np.asarray(box_mean(jnp.asarray(x[None]), width))[0]
    assert out.shape == x.shape
    np.testing.assert_allclose(out, _box_reference(x, width), atol=1e-5)


def test_box_mean_rejects_even_width():
    with pytest.raises(ValueError, match="odd"):
        box_mean(jnp.zeros((1, 8, 2)), 4)


def test_rna_features_values_by_hand():
    rna = np.zeros((1, 20, 2), np.float32)
    rna[0, 10, 0] = 9.0
    rna[0, 3, 1] = 4.0
    feats = np.asarray(rna_features(jnp.asarray(rna)))
    assert feats.shape == (1, 20, NUM_RNA_FEATURES) and feats.dtype == np.float32
    np.testing.assert_allclose(feats[0, :, 0], np.log1p(rna[0, :, 0]), rtol=1e-6)
    np.testing.assert_allclose(feats[0, :, 5], np.log1p(rna[0, :, 1]), rtol=1e-6)
    expected = np.zeros(20)
    expected[6:15] = np.log1p(1.0)  # box9 of a single 9 at position 10
    np.testing.assert_allclose(feats[0, :, 1], expected, atol=1e-6)
    minus9 = np.zeros(20)
    minus9[0:8] = np.log1p(4 / 9)
    np.testing.assert_allclose(feats[0, :, 6], minus9, atol=1e-6)
    np.testing.assert_allclose(feats[0, :, 4], np.log1p(9 / 513), rtol=1e-5)


def test_rna_features_follow_reverse_complement_and_shift():
    """Features of the augmented input equal the augmented features (strand blocks swapped)."""
    rng = np.random.default_rng(0)
    length = 600
    onehot = np.eye(4, dtype=np.float32)[rng.integers(0, 4, length)]
    rna = rng.poisson(1.0, (length, 2)).astype(np.float32)
    sequence = np.concatenate([onehot, rna], axis=1)
    targets = np.zeros((length, 4), np.float32)
    pairs = [(0, 1), (2, 3)]
    seq_rc, _, _ = reverse_complement(sequence, targets, targets > 0, pairs)
    feats = np.asarray(rna_features(jnp.asarray(sequence[None, :, 4:])))[0]
    feats_rc = np.asarray(rna_features(jnp.asarray(seq_rc[None, :, 4:])))[0]
    swapped = np.concatenate([feats[::-1, 5:], feats[::-1, :5]], axis=1)
    np.testing.assert_allclose(feats_rc, swapped, atol=1e-5)
    seq_shift, _, _ = shift(sequence, targets, targets > 0, 7)
    feats_shift = np.asarray(rna_features(jnp.asarray(seq_shift[None, :, 4:])))[0]
    # away from the window edges (wider than the widest box), features shift with the input
    np.testing.assert_allclose(feats_shift[7 + 256 : -256], feats[256 : -256 - 7], atol=1e-5)


def test_rna_features_jit_and_grad_finite():
    rna = jnp.asarray(np.random.default_rng(1).poisson(2.0, (2, 64, 2)).astype(np.float32))
    fn = jax.jit(rna_features)
    np.testing.assert_allclose(np.asarray(fn(rna)), np.asarray(rna_features(rna)), atol=1e-6)
    grad = jax.grad(lambda x: rna_features(x).sum())(rna)
    assert bool(jnp.isfinite(grad).all())


# --- one-sided (strand-aware upstream / downstream) RNA features ---


def _onesided_reference(x: np.ndarray, widths, rise_width, threshold, cap):
    """Plus-strand reference: per position contrasts and distance from the last 5' rise."""
    n = len(x)
    off = max(widths + (rise_width,))
    pad = np.concatenate([np.zeros(off), x, np.zeros(off)])

    def up(i, w):
        return pad[off + i - w : off + i].sum() / w

    def down(i, w):
        return pad[off + i : off + i + w].sum() / w

    out = np.zeros((n, len(widths) + 1))
    last = None
    for i in range(n):
        for k, w in enumerate(widths):
            out[i, k] = np.log1p(down(i, w)) - np.log1p(up(i, w))
        rise = np.log1p(down(i, rise_width)) - np.log1p(up(i, rise_width))
        if rise > threshold:
            last = i
        d = cap if last is None else min(i - last, cap)
        out[i, -1] = np.log1p(d)
    return out


def test_onesided_features_by_reference_on_both_strands():
    from tisiago.tracks.rna_features import (
        NUM_ONESIDED_FEATURES,
        ONESIDED_CAP,
        ONESIDED_RISE_THRESHOLD,
        ONESIDED_RISE_WIDTH,
        ONESIDED_WIDTHS,
        onesided_features,
    )

    rng = np.random.default_rng(4)
    length = 300
    rna = np.zeros((length, 2), np.float32)
    rna[120:, 0] = rng.poisson(5.0, length - 120)  # a plus-strand transcript starting at 120
    rna[:200, 1] = rng.poisson(3.0, 200)  # a minus-strand transcript whose 5' end is at 199
    feats = np.asarray(onesided_features(jnp.asarray(rna[None])))[0]
    half = NUM_ONESIDED_FEATURES // 2
    assert feats.shape == (length, NUM_ONESIDED_FEATURES)
    args = (ONESIDED_WIDTHS, ONESIDED_RISE_WIDTH, ONESIDED_RISE_THRESHOLD, ONESIDED_CAP)
    plus = _onesided_reference(rna[:, 0].astype(np.float64), *args)
    minus = _onesided_reference(rna[::-1, 1].astype(np.float64), *args)[::-1]
    np.testing.assert_allclose(feats[:, :half], plus, atol=1e-4)
    np.testing.assert_allclose(feats[:, half:], minus, atol=1e-4)
    # the distance feature is small just downstream of each 5' rise
    assert feats[125, half - 1] < np.log1p(10) and feats[195, -1] < np.log1p(10)


def test_onesided_features_follow_reverse_complement():
    from tisiago.tracks.rna_features import NUM_ONESIDED_FEATURES, onesided_features

    rng = np.random.default_rng(5)
    length = 400
    onehot = np.eye(4, dtype=np.float32)[rng.integers(0, 4, length)]
    rna = rng.poisson(0.5, (length, 2)).astype(np.float32)
    rna[100:300] += rng.poisson(4.0, (200, 2))
    sequence = np.concatenate([onehot, rna], axis=1)
    targets = np.zeros((length, 4), np.float32)
    seq_rc, _, _ = reverse_complement(sequence, targets, targets > 0, [(0, 1), (2, 3)])
    feats = np.asarray(onesided_features(jnp.asarray(sequence[None, :, 4:])))[0]
    feats_rc = np.asarray(onesided_features(jnp.asarray(seq_rc[None, :, 4:])))[0]
    half = NUM_ONESIDED_FEATURES // 2
    swapped = np.concatenate([feats[::-1, half:], feats[::-1, :half]], axis=1)
    np.testing.assert_allclose(feats_rc, swapped, atol=1e-4)
    grad = jax.grad(lambda x: onesided_features(x).sum())(jnp.asarray(rna[None]))
    assert bool(jnp.isfinite(grad).all())


def test_strand_log_ratios_by_hand_and_reverse_complement():
    from tisiago.tracks.rna_features import strand_log_ratios

    x = np.zeros((1, 12, 2), np.float32)
    x[0, 6:, 0] = 2.0  # plus: rise at 6
    x[0, :4, 1] = 1.0  # minus: 5' end at 3
    out = np.asarray(strand_log_ratios(jnp.asarray(x), (2,), 0.5))[0]
    assert out.shape == (12, 2)
    assert out[6, 0] == pytest.approx(np.log(2.5 / 0.5))  # down mean 2, up mean 0
    assert out[3, 1] == pytest.approx(np.log(1.5 / 0.5))  # minus: down = 3,2 ; up = 4,5
    assert out[2, 0] == pytest.approx(0.0)
    rc = np.asarray(strand_log_ratios(jnp.asarray(x[:, ::-1, ::-1].copy()), (2,), 0.5))[0]
    np.testing.assert_allclose(rc, out[::-1, ::-1], atol=1e-6)
