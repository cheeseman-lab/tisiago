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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts-dir", required=True, help="Directory of *.npz shard partials.")
    ap.add_argument("--glob", required=True, help="Which partials to read (e.g. 'evo2_8k_shard*.npz').")
    ap.add_argument("--exp-store", required=True, help="Compact store with manifest.parquet (src_row_idx).")
    args = ap.parse_args()

    store = Path(args.exp_store)
    manifest = pd.read_parquet(store / "manifest.parquet")
    m = len(manifest)
    assert (manifest.row_idx.values == np.arange(m)).all(), "manifest row_idx must be 0..M-1 contiguous"
    if "src_row_idx" not in manifest.columns:
        raise SystemExit("manifest has no src_row_idx column — cannot map shards to compact rows")

    # src coordinate (0..N-1 in the dense scan) -> compact row, -1 if not wanted
    src = manifest.src_row_idx.values
    src2cmp = np.full(int(src.max()) + 1, -1, dtype=np.int64)
    src2cmp[src] = manifest.row_idx.values

    parts = sorted(Path(args.parts_dir).glob(args.glob))
    if not parts:
        raise SystemExit(f"No partials matching {args.glob!r} in {args.parts_dir}")
    print(f"Found {len(parts)} shards matching {args.glob!r}; gathering {m:,} compact rows", flush=True)

    arrays: dict[str, np.ndarray] = {}
    covered: dict[str, np.ndarray] = {}

    for i, p in enumerate(parts):
        z = np.load(p)
        rows = z["row_idx"]
        # shards cover every scan position; wanted rows are all <= src.max(), so any
        # shard index beyond the map is definitionally not-wanted — mask before indexing.
        cmp = np.full(len(rows), -1, dtype=np.int64)
        in_range = rows < len(src2cmp)
        cmp[in_range] = src2cmp[rows[in_range]]
        keep = cmp >= 0
        dst = cmp[keep]
        for key in z.files:
            if key == "row_idx":
                continue
            mat = z[key]
            if key not in arrays:
                arrays[key] = np.zeros((m, mat.shape[1]), dtype=np.float16)
                covered[key] = np.zeros(m, dtype=bool)
            arrays[key][dst] = mat[keep]
            covered[key][dst] = True
        print(f"  [{i + 1}/{len(parts)}] {p.name}: {int(keep.sum()):,} wanted rows", flush=True)

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
        prov["keys"][key] = {"path": str(out_path.relative_to(store)), "dim": int(mat.shape[1]), "covered": cov}
        flag = "" if cov == m else f"  !! only {cov}/{m} covered"
        print(f"  saved {key} -> {out_path.relative_to(store)}  dim={mat.shape[1]} covered={cov}/{m}{flag}", flush=True)
        assert cov == m, f"{key}: {cov}/{m} rows covered — shards do not cover all wanted rows"

    with open(cfg_path, "w") as f:
        yaml.safe_dump(prov, f, default_flow_style=False, sort_keys=False)
    print(f"Gathered {len(arrays)} keys into {store}", flush=True)


if __name__ == "__main__":
    main()
