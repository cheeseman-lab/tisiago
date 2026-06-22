"""Shared streaming reader over extraction shards (Regime A gather + Regime B scoring)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def build_src_to_compact(manifest: pd.DataFrame) -> np.ndarray:
    """Build int64 map from source row indices to compact row indices.

    Args:
        manifest: DataFrame with columns 'src_row_idx' and 'row_idx'.

    Returns:
        int64 array of length max(src_row_idx)+1, where value is the compact row
        index or -1 if not wanted.
    """
    src = manifest.src_row_idx.values
    src2cmp = np.full(int(src.max()) + 1, -1, dtype=np.int64)
    src2cmp[src] = manifest.row_idx.values
    return src2cmp


def map_rows(
    rows: np.ndarray, src2cmp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map shard row indices to compact indices, masking out-of-range and unwanted.

    Args:
        rows: Array of source row indices from a shard.
        src2cmp: Mapping from source to compact indices (from build_src_to_compact).

    Returns:
        Tuple of (keep_mask, dst) where dst = src2cmp[rows][keep_mask].
        Masks out rows >= len(src2cmp) or where src2cmp[rows] < 0.
    """
    cmp = np.full(len(rows), -1, dtype=np.int64)
    in_range = rows < len(src2cmp)
    cmp[in_range] = src2cmp[rows[in_range]]
    keep = cmp >= 0
    return keep, cmp[keep]


def iter_shards(
    parts_dir: Path, glob: str
) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Iterate over shards, yielding row indices and member arrays.

    Args:
        parts_dir: Directory containing shards.
        glob: Glob pattern to match shard files.

    Yields:
        Tuples of (rows, members) where rows is the row_idx array and members
        is a dict of all other arrays (excluding row_idx).
    """
    for p in sorted(Path(parts_dir).glob(glob)):
        z = np.load(p)
        members = {k: z[k] for k in z.files if k != "row_idx"}
        yield z["row_idx"], members
