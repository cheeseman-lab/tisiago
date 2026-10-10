"""Open-reading-frame features from a one-hot sequence, in JAX (jit-safe, static shapes).

For each strand (minus = computed on the reverse complement, flipped back) and each relative
frame k (position i is base k of the codon starting at i - k), over in-frame codon starts
j <= i - k on that strand:

- ``stop_dist``: log1p(i - j) / log1p(cap) for the nearest TAA/TAG/TGA, distance capped at
  ``cap`` (so in [0, 1]);
- ``atg_dist``: the same for the nearest ATG;
- ``open``: 1 if the nearest in-frame ATG is after the nearest in-frame stop.

A stop codon closes its own positions. Codons that run past the window or contain N are absent,
so window edges read as capped distances. :func:`strand_features_np` is the same definition in
NumPy (one strand of one sequence, no JAX import), for features computed in the data loader.
"""

from __future__ import annotations

import numpy as np

BASES = "ACGT"
STOPS = ("TAA", "TAG", "TGA")
START = "ATG"
DEFAULT_CAP = 3000
FEATURE_NAMES = tuple(
    f"{strand}_{kind}_k{k}"
    for strand in ("plus", "minus")
    for kind in ("stop_dist", "atg_dist", "open")
    for k in range(3)
)
NUM_FEATURES = len(FEATURE_NAMES)


def _jnp():
    import jax.numpy as jnp

    return jnp


def _codon_hits(x, codons, xp):
    """``(B, S)`` bool: a codon from ``codons`` starts at each position."""
    length = x.shape[1]
    padded = xp.pad(x, ((0, 0), (0, 2), (0, 0)))
    hits = xp.zeros(x.shape[:2], bool)
    for codon in codons:
        hit = xp.ones(x.shape[:2], bool)
        for offset, base in enumerate(codon):
            hit = hit & padded[:, offset : offset + length, BASES.index(base)]
        hits = hits | hit
    return hits


def _last_in_frame(hits, xp):
    """``(B, S)`` int32: last hit index j' <= j with j' = j (mod 3), else -1."""
    batch, length = hits.shape
    padded_length = -(-length // 3) * 3
    index = xp.where(hits, xp.arange(length, dtype=xp.int32), -1).astype(xp.int32)
    index = xp.pad(index, ((0, 0), (0, padded_length - length)), constant_values=-1)
    index = index.reshape(batch, padded_length // 3, 3)
    if xp is np:
        index = np.maximum.accumulate(index, axis=1)
    else:
        import jax

        index = jax.lax.cummax(index, axis=1)
    return index.reshape(batch, padded_length)[:, :length]


def _strand_features(x, cap: int, xp):
    """``(B, S, 9)`` features of the plus strand of boolean one-hot ``x``."""
    length = x.shape[1]
    position = xp.arange(length, dtype=xp.int32)
    last = {
        "stop": _last_in_frame(_codon_hits(x, STOPS, xp), xp),
        "atg": _last_in_frame(_codon_hits(x, (START,), xp), xp),
    }
    shifted = {
        name: [xp.pad(value, ((0, 0), (k, 0)), constant_values=-1)[:, :length] for k in range(3)]
        for name, value in last.items()
    }

    def distance(j):
        clipped = xp.where(j >= 0, xp.minimum(position - j, cap), cap).astype(xp.float32)
        return xp.log1p(clipped) / xp.log1p(xp.float32(cap))

    columns = [distance(j) for j in shifted["stop"]] + [distance(j) for j in shifted["atg"]]
    columns += [
        (atg > stop).astype(xp.float32)
        for atg, stop in zip(shifted["atg"], shifted["stop"], strict=True)
    ]
    return xp.stack(columns, axis=-1)


def strand_features_np(onehot: np.ndarray, cap: int = DEFAULT_CAP) -> np.ndarray:
    """``(S, 9)`` float32 plus-strand features of one-hot ``(S, 4)`` (NumPy, no JAX)."""
    return _strand_features(np.asarray(onehot)[None] > 0.5, cap, np)[0].astype(np.float32)


def orf_features(sequence, cap: int = DEFAULT_CAP):
    """``(B, S, 18)`` float32 plus-then-minus open-frame features of one-hot ``(B, S, 4)``."""
    jnp = _jnp()
    x = sequence > 0.5
    plus = _strand_features(x, cap, jnp)
    minus = _strand_features(x[:, ::-1, ::-1], cap, jnp)[:, ::-1]
    return jnp.concatenate([plus, minus], axis=-1)
