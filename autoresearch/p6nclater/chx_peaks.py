"""Real vs model (sbflux s0) CHX peak at reproducible in-CDS (>100 nt) NC starts vs in-frame controls.

Usage (repo root, CPU): python autoresearch/p6nclater/chx_peaks.py
"""

# ruff: noqa
import sys, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, "autoresearch/p6sharp")
from sharpness import export_predictions
from tisiago.seq2func import fastloop
from tisiago.seq2func.txcache import TxCache

for c, l in (("K562", "k562"), ("HeLa", "hela")):
    cache = TxCache(Path(f"data/p6/txcache/fc2d_{l}"))
    k = cache.condition_index(c)
    t = pd.read_parquet(f"data/p6nclater/sites_{l}.parquet")
    t = t[(t.real & t.rep2_hit) | (~t.real)]
    t = t[(t.frame == "cds_inframe") & (t.dist_cds_start > 100) & (t.cls == "nc")]
    preds = export_predictions(cache, Path(f"data/p6/txtrain/export/sbflux_{l}_s0"), c)
    rows = cache.rows("val").set_index("transcript_id")
    res = []
    for tid, g in t.groupby("transcript_id"):
        r = rows.loc[tid]
        lo = int(r.offset)
        hi = lo + int(r.length)
        chx = np.asarray(cache.array(r.shard, "chx")[k, lo:hi, 2], float)
        m = np.clip(preds[tid][k, :, 1], 0, None)
        pos = g.tx_pos.to_numpy(int)
        for name, v in (("real", chx), ("model", m)):
            P = fastloop.site_profiles(v, pos)
            w = P[:, 29:32].sum(1) / np.maximum(P.sum(1), 1e-12)
            res.append(
                pd.DataFrame(
                    {
                        "real": g.real.values,
                        "cls": g.cls.values,
                        "src": name,
                        "share": w,
                        "hit": fastloop.site_hits(P),
                    }
                )
            )
    d = pd.concat(res)
    print(c)
    print(
        d.groupby(["src", "real"])
        .agg(n=("share", "size"), chx_share1=("share", "mean"), chx_argmax=("hit", "mean"))
        .round(3)
    )
