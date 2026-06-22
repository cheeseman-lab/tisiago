"""Build a compact memmap-per-key store from shards, driven by a dataset.yaml."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.dataset_config import DatasetConfig, load_dataset_config, write_provenance
from tisiago.shard_io import build_src_to_compact, iter_shards, map_rows


def build_store(cfg: DatasetConfig, git_sha: str) -> Path:
    """Build a compact memmap-per-key store from shards.

    Args:
        cfg: Dataset configuration specifying parts_dir, out_store, and keys.
        git_sha: Git SHA for provenance tracking.

    Returns:
        Path to the output store directory.
    """
    out = Path(cfg.out_store)
    manifest = pd.read_parquet(out / "manifest.parquet")
    m = len(manifest)
    assert (manifest.row_idx.values == np.arange(m)).all(), "row_idx must be 0..M-1"
    src2cmp = build_src_to_compact(manifest)
    want = [k[:-4].replace("/", "::") for k in cfg.keys]
    arrays, covered = {}, {}
    for rows, members in iter_shards(Path(cfg.parts_dir), "*.npz"):
        keep, dst = map_rows(rows, src2cmp)
        for w in want:
            if w not in members:
                continue
            mat = members[w]
            if w not in arrays:
                arrays[w] = np.zeros((m, mat.shape[1]), dtype=np.float16)
                covered[w] = np.zeros(m, dtype=bool)
            arrays[w][dst] = mat[keep]
            covered[w][dst] = True
    keys_meta = {}
    for w, mat in arrays.items():
        backend, ltag, layer, off = w.split("::")
        op = out / "embeddings" / backend / ltag / layer / f"{off}.npy"
        op.parent.mkdir(parents=True, exist_ok=True)
        np.save(op, mat)
        cov = int(covered[w].sum())
        assert cov == m, f"{w}: {cov}/{m} covered"
        keys_meta[w] = {
            "path": str(op.relative_to(out)),
            "dim": int(mat.shape[1]),
            "covered": cov,
        }
    write_provenance(out, cfg, keys_meta, git_sha)
    return out


def main() -> None:
    """Entry point for building a store from a config file."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    out = build_store(load_dataset_config(args.config), sha)
    print(f"built store at {out}")


if __name__ == "__main__":
    main()
