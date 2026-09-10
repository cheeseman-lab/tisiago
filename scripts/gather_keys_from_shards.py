"""Gather feature keys directly from extraction shards into a compact store.

The dense scan writes per-shard ``.npz`` partials over *every* codon (~62.7M
positions). Assembling those into row-aligned ``[62.7M, D]`` monoliths costs
~514 GB per Evo2 key — wasteful when a head experiment only needs the ~4.3M
candidate rows (train sample + held-out cognate + non-cognate control).

This reads each shard once and scatters only the wanted rows into a compact
``[M, D]`` array, row-aligned to an existing experiment manifest's ``row_idx``
(via its ``src_row_idx`` provenance column = the shard ``row_idx`` coordinate).
No monolith is ever materialised.

Run on a big-memory node: the in-RAM accumulators are ``M x D x len(offsets)``
(~142 GB for the four 4096-d Evo2 offsets at M=4.3M).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tisiago.shard_io import build_src_to_compact, iter_shards, map_rows


def main() -> None:
    """Gather selected shard rows into an existing compact experiment store."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parts-dir", required=True, help="Directory of *.npz shard partials.")
    ap.add_argument(
        "--glob",
        required=True,
        help="Which partials to read (e.g. 'evo2_8k_shard*.npz').",
    )
    ap.add_argument(
        "--exp-store",
        required=True,
        help="Compact store with manifest.parquet (src_row_idx).",
    )
    args = ap.parse_args()

    store = Path(args.exp_store)
    manifest = pd.read_parquet(store / "manifest.parquet")
    m = len(manifest)
    if not (manifest.row_idx.values == np.arange(m)).all():
        raise ValueError("manifest row_idx must be 0..M-1 contiguous")
    if "src_row_idx" not in manifest.columns:
        raise SystemExit("manifest has no src_row_idx column — cannot map shards to compact rows")

    src2cmp = build_src_to_compact(manifest)

    parts = sorted(Path(args.parts_dir).glob(args.glob))
    if not parts:
        raise SystemExit(f"No partials matching {args.glob!r} in {args.parts_dir}")
    print(
        f"Found {len(parts)} shards matching {args.glob!r}; "
        f"gathering {m:,} compact rows",
        flush=True,
    )

    arrays: dict[str, np.ndarray] = {}
    covered: dict[str, np.ndarray] = {}

    for i, (rows, members) in enumerate(iter_shards(Path(args.parts_dir), args.glob)):
        keep, dst = map_rows(rows, src2cmp)
        for key, mat in members.items():
            if key not in arrays:
                arrays[key] = np.zeros((m, mat.shape[1]), dtype=np.float16)
                covered[key] = np.zeros(m, dtype=bool)
            arrays[key][dst] = mat[keep]
            covered[key][dst] = True
        print(f"  shard {i}: {int(keep.sum()):,} wanted rows", flush=True)

    emb_root = store / "embeddings"
    cfg_path = store / "config.yaml"
    prov = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    prov.setdefault("keys", {})
    for key, mat in arrays.items():
        backend, ltag, layer, off = key.split("::")
        out_path = emb_root / backend / ltag / layer / f"{off}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mat)
        cov = int(covered[key].sum())
        prov["keys"][key] = {
            "path": str(out_path.relative_to(store)),
            "dim": int(mat.shape[1]),
            "covered": cov,
        }
        flag = "" if cov == m else f"  !! only {cov}/{m} covered"
        print(
            f"  saved {key} -> {out_path.relative_to(store)}  "
            f"dim={mat.shape[1]} covered={cov}/{m}{flag}",
            flush=True,
        )
        if cov != m:
            raise ValueError(
                f"{key}: {cov}/{m} rows covered — shards do not cover all wanted rows"
            )

    with open(cfg_path, "w") as f:
        yaml.safe_dump(prov, f, default_flow_style=False, sort_keys=False)
    print(f"Gathered {len(arrays)} keys into {store}", flush=True)


if __name__ == "__main__":
    main()
