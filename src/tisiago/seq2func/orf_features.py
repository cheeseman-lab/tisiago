"""Open-reading-frame features from a one-hot sequence, in JAX (jit-safe, static shapes).

For each strand (minus = computed on the reverse complement, flipped back) and each relative
frame k (position i is base k of the codon starting at i - k), over in-frame codon starts
j <= i - k on that strand:

- ``stop_dist``: log1p(i - j) for the nearest TAA/TAG/TGA, capped at ``cap``;
- ``atg_dist``: log1p(i - j) for the nearest ATG, capped at ``cap``;
- ``open``: 1 if the nearest in-frame ATG is after the nearest in-frame stop.

A stop codon closes its own positions. Codons that run past the window or contain N are absent,
so window edges read as capped distances.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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


def _codon_hits(x, codons):
    """``(B, S)`` bool: a codon from ``codons`` starts at each position."""
    length = x.shape[1]
    padded = jnp.pad(x, ((0, 0), (0, 2), (0, 0)))
    hits = jnp.zeros(x.shape[:2], bool)
    for codon in codons:
        hit = jnp.ones(x.shape[:2], bool)
        for offset, base in enumerate(codon):
            hit = hit & padded[:, offset : offset + length, BASES.index(base)]
        hits = hits | hit
    return hits


def _last_in_frame(hits):
    """``(B, S)`` int32: last hit index j' <= j with j' = j (mod 3), else -1."""
    batch, length = hits.shape
    padded_length = -(-length // 3) * 3
    index = jnp.where(hits, jnp.arange(length, dtype=jnp.int32), -1)
    index = jnp.pad(index, ((0, 0), (0, padded_length - length)), constant_values=-1)
    index = jax.lax.cummax(index.reshape(batch, padded_length // 3, 3), axis=1)
    return index.reshape(batch, padded_length)[:, :length]


def _strand_features(x, cap: int):
    """``(B, S, 9)`` features of the plus strand of boolean one-hot ``x``."""
    length = x.shape[1]
    position = jnp.arange(length, dtype=jnp.int32)
    last = {
        "stop": _last_in_frame(_codon_hits(x, STOPS)),
        "atg": _last_in_frame(_codon_hits(x, (START,))),
    }
    shifted = {
        name: [jnp.pad(value, ((0, 0), (k, 0)), constant_values=-1)[:, :length] for k in range(3)]
        for name, value in last.items()
    }

    def distance(j):
        return jnp.log1p(jnp.where(j >= 0, jnp.minimum(position - j, cap), cap).astype(jnp.float32))

    columns = [distance(j) for j in shifted["stop"]] + [distance(j) for j in shifted["atg"]]
    columns += [
        (atg > stop).astype(jnp.float32)
        for atg, stop in zip(shifted["atg"], shifted["stop"], strict=True)
    ]
    return jnp.stack(columns, axis=-1)


def orf_features(sequence, cap: int = DEFAULT_CAP):
    """``(B, S, 18)`` float32 plus-then-minus open-frame features of one-hot ``(B, S, 4)``."""
    x = sequence > 0.5
    plus = _strand_features(x, cap)
    minus = _strand_features(x[:, ::-1, ::-1], cap)[:, ::-1]
    return jnp.concatenate([plus, minus], axis=-1)
