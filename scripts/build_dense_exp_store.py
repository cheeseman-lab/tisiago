"""Build a compact dense **experiment store** for autoresearch / fast head iteration.

The full dense scan store is 62.7M rows × big keys on NFS — re-reading it per experiment
costs ~30 min. This gathers a *fixed experiment subset* of rows once and writes them in the
same per-key ``.npy`` layout as the curated store, so `dense_caller`/autoresearch load it
instantly and can sweep feature subsets, heads, and (down-)sampled imbalance freely.

Row set (cognate = AUG + near_cognate; non-cognate stays an eval-only grounding control):
  * train: all positives + a capped negative sample (the genome-scale negative diversity)
  * val:   all cognate (calibration + val-metric climbing)
  * test:  all cognate (recall metric) + a non-cognate sample (grounding)

Re-indexed ``row_idx`` 0..M-1. Evo2 keys can be appended for the SAME rows once its scan
store is assembled (pass ``--keys evo2/...`` against the same ``--out-store``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AG_KEYS = [
    "alphagenome_jax/L16k/decoder_1bp/off0.npy",
    "alphagenome_jax/L131k/decoder_1bp/off0.npy",
    "onehot/kozakW20.npy",
]


def main() -> None:
    """Gather the experiment row subset's features into a compact local store."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-store", default="data/scan_store_allsplits")
    ap.add_argument("--out-store", default="data/dense_exp_store")
    ap.add_argument("--keys", nargs="+", default=AG_KEYS)
    ap.add_argument("--neg-cap", type=int, default=2_000_000)
    ap.add_argument("--noncog-sample", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    scan = Path(args.scan_store)
    emb = scan / "embeddings"
    out = Path(args.out_store)
    (out / "embeddings").mkdir(parents=True, exist_ok=True)

    # If the manifest already exists (a prior key-append run), reuse its exact rows so all
    # keys stay row-aligned; otherwise select the experiment rows fresh.
    if (out / "manifest.parquet").exists():
        em = pd.read_parquet(out / "manifest.parquet")
        rows = em["src_row_idx"].values
        print(f"reusing existing experiment rows: {len(rows):,}", flush=True)
    else:
        m = pd.read_parquet(scan / "manifest.parquet")
        sp, y = m.split.values, m.label_tis.values
        cog = m.codon_class.isin(["AUG", "near_cognate"]).values
        noncog = m.codon_class.values == "non_cognate"
        tn = np.where((sp == "train") & cog & (y == 0))[0]
        if len(tn) > args.neg_cap:
            tn = rng.choice(tn, args.neg_cap, replace=False)
        tnc = np.where((sp == "test") & noncog)[0]
        if len(tnc) > args.noncog_sample:
            tnc = rng.choice(tnc, args.noncog_sample, replace=False)
        rows = np.sort(np.concatenate([
            np.where((sp == "train") & cog & (y == 1))[0], tn,
            np.where((sp == "val") & cog)[0],
            np.where((sp == "test") & cog)[0], tnc,
        ]))
        em = m.iloc[rows].reset_index(drop=True).copy()
        em["src_row_idx"] = rows            # provenance back to the full scan manifest
        em["row_idx"] = np.arange(len(em))  # 0..M-1 for the experiment store
        em.to_parquet(out / "manifest.parquet", index=False)
        print(f"experiment rows: {len(rows):,}  splits={em.split.value_counts().to_dict()}",
              flush=True)

    # SEQUENTIAL read: stream the whole source array in big blocks and keep the wanted rows.
    # `rows` is sorted, so masked extraction in source order matches the manifest order.
    # Reading 193GB start-to-finish (~6 min) beats millions of random fancy-index seeks (~85 min).
    for k in args.keys:
        a = np.load(emb / k, mmap_mode="r")
        n_src = a.shape[0]
        mask = np.zeros(n_src, dtype=bool)
        mask[rows] = True
        mat = np.empty((len(rows), a.shape[1]), dtype=np.float16)
        op, block = 0, 4_000_000
        for s in range(0, n_src, block):
            e = min(s + block, n_src)
            bm = mask[s:e]
            cnt = int(bm.sum())
            if cnt:
                mat[op : op + cnt] = np.asarray(a[s:e])[bm]
                op += cnt
            print(f"  {k}: read {e:,}/{n_src:,}  kept {op:,}/{len(rows):,}", flush=True)
        if op != len(rows):
            raise ValueError(f"collected {op} != expected {len(rows)}")
        outp = out / "embeddings" / k
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.save(outp, mat)
        print(f"  saved {k}  {mat.shape}", flush=True)
    print(f"dense experiment store at {out}", flush=True)


if __name__ == "__main__":
    main()
