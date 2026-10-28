"""Top-1% TIS concentration per condition (rep1, rep2) at a common per-transcript depth, val shard.

Usage: python autoresearch/p6sharp/line_concentration.py [CACHE] — CPU, seconds. Transcripts with >= DEPTH reads in
both replicates; each replicate thinned to DEPTH reads (multinomial), share of reads in its top 1% positions.
"""

import json
import sys

import numpy as np
import pandas as pd

DEPTH = 50
cache = sys.argv[1] if len(sys.argv) > 1 else "data/p6/txcache/fc2d_k562"
conds = json.load(open(f"{cache}/cache.json"))["conditions"]
idx = pd.read_parquet(f"{cache}/index.parquet")
idx = idx[idx.shard == "val_chr7"]
tis = np.load(f"{cache}/shards/val_chr7/tis.npy", mmap_mode="r")
rng = np.random.default_rng(0)
for k, c in enumerate(conds):
    out = {0: [], 1: []}
    for row in idx.itertuples():
        x = np.asarray(tis[k, row.offset : row.offset + row.length, :2], np.float64)
        tot = x.sum(0)
        if (tot < DEPTH).any():
            continue
        top = max(1, int(round(0.01 * row.length)))
        for r in (0, 1):
            s = rng.multinomial(DEPTH, x[:, r] / tot[r])
            out[r].append(np.sort(s)[-top:].sum() / DEPTH)
    print(f"{c:12s} n={len(out[0]):4d} top1 rep1 {np.mean(out[0]):.3f} rep2 {np.mean(out[1]):.3f}")
