"""Per-condition loss mask: blank Ribo-seq over unexpressed genes is missing data, not zero."""

from __future__ import annotations

import numpy as np
import pandas as pd

MASKED_LIBRARIES = ("tis", "chx")


def expression_mask(layout, genes: pd.DataFrame, chrom: str, start: int, end: int) -> np.ndarray:
    """Boolean ``(end - start, C)`` mask; False = excluded from the loss."""
    mask = np.ones((end - start, len(layout)), dtype=bool)
    local = genes[(genes.chrom == chrom) & (genes.end > start) & (genes.start < end)]
    for index, (condition, library, strand) in enumerate(layout):
        if library not in MASKED_LIBRARIES:
            continue
        rows = local[(local.condition == condition) & (local.strand == strand)]
        silent = np.zeros(end - start, dtype=bool)
        for row in rows[~rows.expressed].itertuples():
            silent[max(row.start, start) - start : min(row.end, end) - start] = True
        for row in rows[rows.expressed].itertuples():
            silent[max(row.start, start) - start : min(row.end, end) - start] = False
        mask[:, index] = ~silent
    return mask
