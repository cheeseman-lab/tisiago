"""Memory-bounded, atomic output assembly for embedding-extraction shards."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


class PreallocatedShard:
    """Collect fixed-width candidate features without per-vector Python objects.

    Rows are stored in sorted source-row order. Feature matrices are allocated lazily
    when their width is first observed, then filled by vectorized row assignment.
    """

    def __init__(self, row_idx, *, dtype=np.float16) -> None:
        """Initialize storage for a sorted, unique set of source row indices."""
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        if self.row_idx.ndim != 1:
            raise ValueError("row_idx must be one-dimensional")
        if len(self.row_idx) and not np.all(self.row_idx[1:] > self.row_idx[:-1]):
            raise ValueError("row_idx must be sorted and unique")
        self.dtype = np.dtype(dtype)
        self.arrays: dict[str, np.ndarray] = {}
        self.covered: dict[str, np.ndarray] = {}

    def add(self, key: str, row_idx, values) -> None:
        """Add one batch of rows for a feature key."""
        rows = np.asarray(row_idx, dtype=np.int64)
        values = np.asarray(values)
        if rows.ndim != 1 or values.ndim != 2 or values.shape[0] != len(rows):
            raise ValueError("values must have shape (len(row_idx), feature_width)")
        slots = np.searchsorted(self.row_idx, rows)
        valid = slots < len(self.row_idx)
        if not valid.all() or not np.array_equal(self.row_idx[slots], rows):
            raise ValueError("batch contains row indices outside this shard")

        if key not in self.arrays:
            self.arrays[key] = np.empty((len(self.row_idx), values.shape[1]), dtype=self.dtype)
            self.covered[key] = np.zeros(len(self.row_idx), dtype=bool)
        elif self.arrays[key].shape[1] != values.shape[1]:
            raise ValueError(f"{key}: feature width changed between batches")
        if self.covered[key][slots].any():
            raise ValueError(f"{key}: duplicate output rows")

        self.arrays[key][slots] = values
        self.covered[key][slots] = True

    def validate(self, expected_keys: set[str] | None = None) -> None:
        """Raise when keys or candidate rows are missing."""
        if expected_keys is not None:
            missing = expected_keys - self.arrays.keys()
            unexpected = self.arrays.keys() - expected_keys
            if missing or unexpected:
                raise ValueError(
                    f"output-key mismatch: missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
        for key, mask in self.covered.items():
            n_covered = int(mask.sum())
            if n_covered != len(self.row_idx):
                raise ValueError(f"{key}: covered {n_covered}/{len(self.row_idx)} rows")
            if not np.isfinite(self.arrays[key]).all():
                raise ValueError(f"{key}: output contains non-finite values")

    def save(
        self,
        path: str | Path,
        *,
        expected_keys: set[str] | None = None,
        compressed: bool = False,
    ) -> None:
        """Validate and atomically write the shard as an NPZ archive."""
        self.validate(expected_keys)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"row_idx": self.row_idx, **self.arrays}
        writer = np.savez_compressed if compressed else np.savez
        tmp = tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        )
        tmp_path = Path(tmp.name)
        try:
            with tmp:
                writer(tmp, **payload)
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


def shard_is_complete(path: str | Path, row_idx, expected_keys: set[str]) -> bool:
    """Return whether an existing atomic shard has the expected rows and members."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(path) as shard:
            if set(shard.files) != {"row_idx", *expected_keys}:
                return False
            return np.array_equal(shard["row_idx"], np.asarray(row_idx, dtype=np.int64))
    except (OSError, ValueError, KeyError):
        return False
