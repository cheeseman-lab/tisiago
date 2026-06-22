"""Assemble per-shard extraction partials into a row-aligned vector store.

Reads every ``*.npz`` partial written by ``extract_candidates.py`` and scatters
its candidate vectors into global ``[N, D]`` float16 arrays, one ``.npy`` per
``backend × length × layer × offset``, row-aligned to ``manifest.row_idx`` (so
row ``i`` is the candidate with ``row_idx == i``).

A head experiment is then: load a few ``.npy``, ``np.concatenate`` on axis 1,
filter rows by the manifest ``split`` column. No GPU.

Layout::

    store/
      manifest.parquet
      config.yaml
      embeddings/
        alphagenome_jax/L16k/decoder_1bp/off0.npy        # [N, 1536] fp16
        alphagenome_jax/L131k/decoder_1bp/off0.npy
        evo2/W8k/blocks.28.mlp.l3/off6.npy               # [N, 4096] fp16
        ...

Run in any env with numpy + pandas + pyyaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--parts-dir", required=True, help="Directory of *.npz shard partials.")
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--glob", default="*.npz",
                    help="Which partials to assemble (e.g. 'ag16k_shard*.npz' for one spec). "
                         "Lets you assemble a subset/one key at a time to bound RAM.")
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    n = len(manifest)
    assert (manifest.row_idx.values == np.arange(n)).all(), "manifest row_idx must be 0..N-1 contiguous"

    parts = sorted(Path(args.parts_dir).glob(args.glob))
    if not parts:
        raise SystemExit(f"No partials matching {args.glob!r} in {args.parts_dir}")
    print(f"Found {len(parts)} shard partials matching {args.glob!r}")

    store = Path(args.store_dir)
    emb_root = store / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    # key -> memmapped/in-RAM (N, D) accumulator, and per-key coverage count
    arrays: dict[str, np.ndarray] = {}
    covered: dict[str, np.ndarray] = {}

    for p in parts:
        z = np.load(p)
        rows = z["row_idx"]
        for key in z.files:
            if key == "row_idx":
                continue
            mat = z[key]
            if key not in arrays:
                arrays[key] = np.zeros((n, mat.shape[1]), dtype=np.float16)
                covered[key] = np.zeros(n, dtype=bool)
            arrays[key][rows] = mat
            covered[key][rows] = True

    provenance = {"n_candidates": int(n), "n_shards": len(parts), "keys": {}}
    for key, mat in arrays.items():
        backend, ltag, layer, off = key.split("::")
        out_path = emb_root / backend / ltag / layer / f"{off}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mat)
        cov = int(covered[key].sum())
        provenance["keys"][key] = {"path": str(out_path.relative_to(store)), "dim": int(mat.shape[1]), "covered": cov}
        flag = "" if cov == n else f"  !! only {cov}/{n} covered"
        print(f"  {key:48s} -> {out_path.relative_to(store)}  dim={mat.shape[1]} covered={cov}/{n}{flag}")

    manifest.to_parquet(store / "manifest.parquet", index=False)
    cfg_path = store / "config.yaml"
    if cfg_path.exists():  # merge keys from prior per-spec runs into the same store
        prior = yaml.safe_load(cfg_path.read_text()) or {}
        merged = {**prior.get("keys", {}), **provenance["keys"]}
        provenance["keys"] = merged
        provenance["n_shards"] = prior.get("n_shards", 0) + len(parts)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(provenance, f, default_flow_style=False, sort_keys=False)
    print(f"Store assembled at {store} ({len(arrays)} feature arrays)")


if __name__ == "__main__":
    main()
