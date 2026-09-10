"""Validation and row-selection helpers for candidate manifests."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

SITE_KEY = ("transcript_id", "mrna_index")


def transcript_shard_mask(transcript_id, shard_id: int, n_shards: int) -> np.ndarray:
    """Assign whole transcripts to deterministic crc32 shards."""
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    if not 0 <= shard_id < n_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < n_shards")
    unique = transcript_id.unique()
    assignment = {
        value: zlib.crc32(str(value).encode()) % n_shards for value in unique
    }
    return transcript_id.map(assignment).to_numpy() == shard_id


def unique_site_indices(manifest: pd.DataFrame) -> np.ndarray:
    """Return one row index per transcript-relative candidate site.

    The historical curated manifest contains repeated positive rows from separate
    cell-line calls. Frozen sequence features are identical for these repetitions,
    so silently retaining them up-weights some positives. This helper validates that
    duplicates agree on the shared biological identity and selects one row.
    """
    required = {*SITE_KEY, "label_tis", "split", "chrom", "gstart", "strand", "codon"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    duplicated = manifest.duplicated(list(SITE_KEY), keep=False)
    if duplicated.any():
        repeated = manifest.loc[duplicated]
        invariant = ("label_tis", "split", "chrom", "gstart", "strand", "codon")
        conflicts = repeated.groupby(list(SITE_KEY), sort=False)[list(invariant)].nunique()
        bad = conflicts.gt(1).any(axis=1)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} duplicate transcript sites have conflicting identity or labels"
            )
    return np.flatnonzero(~manifest.duplicated(list(SITE_KEY), keep="first"))


def split_grouped_indices(indices, groups, *, fraction: float = 0.5, seed: int = 0):
    """Deterministically partition row indices without splitting a group.

    This is used to keep probability calibration and operating-threshold selection
    disjoint while preserving transcript-level grouping.
    """
    indices = np.asarray(indices, dtype=np.int64)
    groups = np.asarray(groups)
    if indices.ndim != 1 or groups.shape != indices.shape:
        raise ValueError("indices and groups must be aligned one-dimensional arrays")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between zero and one")
    modulus = 10_000
    cutoff = round(fraction * modulus)
    first = np.fromiter(
        (
            zlib.crc32(f"{seed}:{group}".encode()) % modulus < cutoff
            for group in groups
        ),
        dtype=bool,
        count=len(groups),
    )
    if not first.any() or first.all():
        raise ValueError("group split produced an empty partition")
    return indices[first], indices[~first]
