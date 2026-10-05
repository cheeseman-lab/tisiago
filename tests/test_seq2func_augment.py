"""Tests for strand-aware augmentation."""

import numpy as np

from tisiago.seq2func.augment import reverse_complement, shift

PAIRS = [(0, 1)]


def _window():
    onehot = np.eye(4, dtype=np.float32)[[0, 1, 2, 3, 0]]  # A C G T A
    targets = np.zeros((5, 2), np.float32)
    targets[1, 0] = 7  # plus P-site at 1
    mask = np.ones((5, 2), bool)
    mask[4, 1] = False
    return onehot, targets, mask


def test_reverse_complement_maps_plus_to_minus_and_back():
    onehot, targets, mask = _window()
    rc_x, rc_y, rc_m = reverse_complement(onehot, targets, mask, PAIRS)
    assert rc_x.argmax(1).tolist() == [3, 0, 1, 2, 3]  # revcomp(ACGTA) = TACGT
    assert rc_y[3, 1] == 7 and rc_y[:, 0].sum() == 0  # 5-1-1 = 3, now minus
    assert not rc_m[0, 0] and rc_m[0, 1]  # mask follows its channel
    back = reverse_complement(rc_x, rc_y, rc_m, PAIRS)
    for a, b in zip(back, (onehot, targets, mask), strict=True):
        np.testing.assert_array_equal(a, b)


def test_shift_moves_everything_without_wrapping():
    onehot, targets, mask = _window()
    x, y, m = shift(onehot, targets, mask, 2)
    assert y[3, 0] == 7
    assert x[:2].sum() == 0 and not m[:2].any() and y[:2].sum() == 0
    x, y, m = shift(onehot, targets, mask, -2)
    assert y[:, 0].sum() == 0  # the P-site at 1 fell off the left edge
    assert x[3:].sum() == 0 and not m[3:].any()
