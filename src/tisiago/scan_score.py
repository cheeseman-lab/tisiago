"""Regime B: score every wanted position by streaming shards — no monolith assembly.

Streams each shard once, concatenates the requested feature keys for its wanted rows into a
preallocated fp32 block, applies the calibrated head, and scatters the scalar predictions into
a result aligned to ``manifest.row_idx``. The full ``[N_scan, D]`` store is never materialized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.shard_io import build_src_to_compact, iter_shards, map_rows


def score_shards(
    parts_dir: Path,
    glob: str,
    manifest: pd.DataFrame,
    keys: list[str],
    head: dict,
    chunk: int = 100_000,
) -> np.ndarray:
    """Score every wanted position by streaming shards once.

    Concatenates the requested feature keys per shard (preallocated, fp32) and applies
    the calibrated head, scattering scalar predictions into a result aligned to
    ``manifest.row_idx``. The full ``[N_scan, D]`` store is never materialized.

    Args:
        parts_dir: Directory containing shards (npz files).
        glob: Glob pattern to match shard files (e.g., "*.npz").
        manifest: DataFrame with columns 'src_row_idx' and 'row_idx'.
        keys: List of feature keys to concatenate (manifest-style paths with .npy).
        head: Fitted head dict with a "predict" callable returning P.
        chunk: Chunk size (unused, kept for interface compatibility).

    Returns:
        np.ndarray of shape (len(manifest),) with probabilities aligned to manifest.row_idx.

    Raises:
        AssertionError: If any wanted row is never seen across shards.
    """
    # member names carry "::" and no ".npy"; manifest keys are slash paths with ".npy"
    want = [
        k[:-4].replace("/", "::") if k.endswith(".npy") else k.replace("/", "::")
        for k in keys
    ]
    m = len(manifest)
    src2cmp = build_src_to_compact(manifest)
    p = np.full(m, np.nan, dtype=np.float64)
    covered = np.zeros(m, dtype=bool)
    for rows, members in iter_shards(Path(parts_dir), glob):
        keep, dst = map_rows(rows, src2cmp)
        if not keep.any():
            continue
        widths = [members[w].shape[1] for w in want]
        X = np.empty((int(keep.sum()), sum(widths)), dtype=np.float32)
        c = 0
        for w, wid in zip(want, widths):
            X[:, c : c + wid] = members[w][keep]
            c += wid
        p[dst] = head["predict"](X)
        covered[dst] = True
    n_cov = int(covered.sum())
    assert (
        n_cov == m
    ), f"covered {n_cov}/{m} rows — shards do not cover all wanted positions"
    return p
