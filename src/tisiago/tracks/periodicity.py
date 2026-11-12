"""CHX codon-phase helpers used by the scorecard (pure numpy)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def region_positions(strand: str, intervals: Sequence[tuple[int, int]]):
    """Genomic positions in transcript order and the exon rank of each (0 = 5'-most)."""
    exons = sorted(intervals, reverse=strand == "-")
    positions, ranks = [], []
    for rank, (start, end) in enumerate(exons):
        span = range(start, end) if strand == "+" else range(end - 1, start - 1, -1)
        positions.extend(span)
        ranks.extend([rank] * (end - start))
    return np.asarray(positions, dtype=np.int64), np.asarray(ranks, dtype=np.int64)


def phase_counts(values: np.ndarray, block: int = 1) -> np.ndarray:
    """``(n_blocks, 3)`` per-phase sums over consecutive blocks of ``block`` codons."""
    values = np.asarray(values, dtype=np.float64)
    n = (len(values) // (3 * block)) * 3 * block
    return values[:n].reshape(-1, block, 3).sum(axis=1)


def dominant_phases(values: np.ndarray, block: int, min_counts: float = 0.0) -> np.ndarray:
    """Dominant phase per block; -1 where the block total is below ``min_counts`` or tied."""
    counts = phase_counts(values, block)
    top = counts.max(axis=1)
    unique = (counts == top[:, None]).sum(axis=1) == 1
    return np.where(
        unique & (counts.sum(axis=1) >= min_counts) & (top > 0), counts.argmax(axis=1), -1
    )
