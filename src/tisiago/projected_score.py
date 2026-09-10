"""Complete calibrated inference from additive model-specific partial logits."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.linear_head import LinearHeadArtifact
from tisiago.shard_io import iter_shards

TXP_PREFIX = "evo2/TXP/"


def assemble_partial_logits(
    parts_dir: Path,
    glob: str,
    n_rows: int,
    artifact_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Assemble projected transcript shards into aligned additive-logit vectors."""
    if n_rows < 0 or not artifact_names:
        raise ValueError("n_rows must be non-negative and artifact_names non-empty")
    partials = {
        name: np.empty(n_rows, dtype=np.float32) for name in artifact_names
    }
    covered = np.zeros(n_rows, dtype=bool)
    n_shards = 0
    expected = {f"partial_logit::{name}" for name in artifact_names}
    for rows, members in iter_shards(parts_dir, glob):
        n_shards += 1
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) and (rows.min() < 0 or rows.max() >= n_rows):
            raise ValueError("projected shard contains out-of-range row indices")
        if covered[rows].any():
            raise ValueError("projected shards contain duplicate row indices")
        if set(members) != expected:
            missing = sorted(expected - members.keys())
            extra = sorted(members.keys() - expected)
            raise ValueError(
                f"projected shard members differ: missing={missing}, extra={extra}"
            )
        for name in artifact_names:
            values = np.asarray(members[f"partial_logit::{name}"])
            if values.shape == (len(rows), 1):
                values = values[:, 0]
            if values.shape != (len(rows),) or not np.isfinite(values).all():
                raise ValueError(f"{name}: invalid projected logits {values.shape}")
            partials[name][rows] = values
        covered[rows] = True
    if not n_shards:
        raise FileNotFoundError(f"no projected shards match {glob!r} in {parts_dir}")
    n_covered = int(covered.sum())
    if n_covered != n_rows:
        raise ValueError(f"projected shards cover {n_covered}/{n_rows} rows")
    return partials


def score_projected_blocks(
    artifacts: Mapping[str, LinearHeadArtifact],
    txp_partials: Mapping[str, np.ndarray],
    embedding_root: Path,
    *,
    chunk_size: int = 100_000,
) -> dict[str, np.ndarray]:
    """Add non-TXP feature blocks and return calibrated per-head probabilities."""
    if set(artifacts) != set(txp_partials):
        raise ValueError("artifact and partial-logit names differ")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    lengths = {len(np.asarray(values)) for values in txp_partials.values()}
    if len(lengths) != 1:
        raise ValueError("partial-logit arrays have different lengths")
    n_rows = lengths.pop()
    external_keys = {
        key
        for artifact in artifacts.values()
        for key in artifact.feature_keys
        if not key.startswith(TXP_PREFIX)
    }
    arrays = {}
    for key in external_keys:
        path = embedding_root / key
        if not path.is_file():
            raise FileNotFoundError(f"missing external feature array: {path}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or len(array) != n_rows:
            raise ValueError(f"{key}: expected ({n_rows}, width), got {array.shape}")
        arrays[key] = array

    predictions = {
        name: np.empty(n_rows, dtype=np.float64) for name in artifacts
    }
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        for name, artifact in artifacts.items():
            partial = np.asarray(txp_partials[name])
            logits = np.asarray(partial[start:end], dtype=np.float32).copy()
            logits += artifact.intercept
            slices = artifact.feature_slices()
            for key in artifact.feature_keys:
                if key.startswith(TXP_PREFIX):
                    continue
                values = np.asarray(arrays[key][start:end], dtype=np.float32)
                expected_width = artifact.feature_dims[artifact.feature_keys.index(key)]
                if values.shape[1] != expected_width:
                    raise ValueError(
                        f"{key}: artifact width {expected_width}, array width {values.shape[1]}"
                    )
                logits += values @ artifact.coefficient[slices[key]]
            predictions[name][start:end] = artifact.calibrate_logits(logits)
    return predictions


def _atomic_save(path: Path, values: np.ndarray) -> None:
    """Atomically save one NumPy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            np.save(temporary, values)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    """Score projected TXP shards plus row-aligned non-TXP feature arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--glob", default="evo2_txp_logits*.npz")
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--head-artifact", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--ensemble-threshold", type=float, default=None)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest, columns=["row_idx"])
    if not np.array_equal(manifest.row_idx.to_numpy(), np.arange(len(manifest))):
        raise ValueError("manifest row_idx must be contiguous and row-aligned")
    artifacts = {}
    for artifact_path in args.head_artifact:
        name = Path(artifact_path).stem
        if name in artifacts:
            raise ValueError(f"duplicate head artifact name: {name}")
        artifacts[name] = LinearHeadArtifact.load(artifact_path)
    partials = assemble_partial_logits(
        Path(args.parts_dir), args.glob, len(manifest), tuple(artifacts)
    )
    predictions = score_projected_blocks(
        artifacts,
        partials,
        Path(args.embedding_root),
        chunk_size=args.chunk_size,
    )
    out_dir = Path(args.out_dir)
    for name, probability in predictions.items():
        _atomic_save(out_dir / f"{name}.npy", probability)
    ensemble = np.mean(list(predictions.values()), axis=0)
    _atomic_save(out_dir / "ensemble.npy", ensemble)
    metadata = {
        "schema_version": 1,
        "manifest": str(Path(args.manifest).resolve()),
        "artifact_names": list(artifacts),
        "ensemble_threshold": args.ensemble_threshold,
        "n_rows": len(manifest),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {len(predictions)} heads plus ensemble for {len(manifest):,} rows")


if __name__ == "__main__":
    main()
