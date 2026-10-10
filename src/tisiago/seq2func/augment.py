"""Strand-aware reverse-complement and shift augmentation for stranded track windows."""

from __future__ import annotations

import numpy as np


def reverse_complement(onehot, targets, mask, pairs):
    """Reverse-complement a window; swap each (plus, minus) channel pair.

    ``onehot`` columns past the first 4 are two stranded feature blocks (plus, then minus, e.g.
    ``tx_features``): they are reversed and the blocks swapped.
    """
    order = np.arange(targets.shape[1])
    for plus, minus in pairs:
        order[plus], order[minus] = minus, plus
    sequence = onehot[::-1, ::-1]
    if onehot.shape[1] > 4:
        extra = onehot[::-1, 4:]
        if extra.shape[1] % 2:
            raise ValueError(
                f"extra sequence channels must be two strand blocks, not {extra.shape}"
            )
        half = extra.shape[1] // 2
        sequence = np.concatenate([onehot[::-1, 3::-1], extra[:, half:], extra[:, :half]], axis=1)
    return sequence.copy(), targets[::-1][:, order].copy(), mask[::-1][:, order].copy()


def shift(onehot, targets, mask, offset: int):
    """Shift content by ``offset`` bases; vacated positions become N / zero / masked."""
    if offset == 0:
        return onehot.copy(), targets.copy(), mask.copy()
    out = (np.zeros_like(onehot), np.zeros_like(targets), np.zeros_like(mask))
    src = slice(0, -offset) if offset > 0 else slice(-offset, None)
    dst = slice(offset, None) if offset > 0 else slice(0, offset)
    for array, moved in zip((onehot, targets, mask), out, strict=True):
        moved[dst] = array[src]
    return out
