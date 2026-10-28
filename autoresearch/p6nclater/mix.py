"""Post-hoc check: TIS' = TIS + alpha * CHX (per-transcript normalised) on reproducible starts.

Tests whether a harringtonine "elongation leak" term (the TIS track also holds a share of the
elongating-ribosome profile) would recover the internal in-frame near-cognate starts missed by
the model, and at what cost to ATG / other starts. Model CHX = the run's own exported CHX; the
``real`` CHX source (pooled replicates) is an oracle bound. Hit = argmax within 1 nt over +-30 nt,
conditioned on replicate 2 (``fastloop.rep_summary``). Validation only; no training.

Usage:
    python autoresearch/p6nclater/mix.py --cache data/p6/txcache/fc2d_k562 --condition K562 \
        --runs sbflux_k562_s0 sbflux_k562_s1 sbflux_k562_s2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func import fastloop
from tisiago.seq2func.txcache import TxCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p6sharp"))
from sharpness import export_predictions  # noqa: E402

ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def site_group(s) -> str:
    """Mutually exclusive start category (MANE CDS used for analysis only)."""
    if str(s.codon).upper() == "ATG":
        return "atg_" + s.region
    if s.frame == "cds_inframe" and s.dist_cds_start > 100:
        return "nc_cds>100"
    if s.frame == "cds_inframe" and s.dist_cds_start > 0:
        return "nc_cds<=100"
    return "nc_noncds_" + s.region


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sites", type=Path, default=Path("data/p6nclater"))
    ap.add_argument("--export", type=Path, default=Path("data/p6/txtrain/export"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "mix.tsv")
    args = ap.parse_args()
    cache = TxCache(args.cache)
    if cache.config.get("split") == "test":
        raise ValueError("validation only")
    c = args.condition
    k = cache.condition_index(c)
    table = pd.read_parquet(args.sites / f"sites_{c.lower()}.parquet")
    table = table[table.real & table.rep2_hit].copy()
    table["group"] = [site_group(s) for s in table.itertuples()]
    rows = cache.rows("val").set_index("transcript_id")
    preds = {run: export_predictions(cache, args.export / run, c) for run in args.runs}
    hits = {}
    for tid, group in table.groupby("transcript_id", sort=False):
        row = rows.loc[tid]
        lo, hi = int(row.offset), int(row.offset) + int(row.length)
        real_chx = np.asarray(cache.array(row.shard, "chx")[k, lo:hi, 2], np.float64)
        pos = group.tx_pos.to_numpy(int)
        for run in args.runs:
            p = np.clip(np.asarray(preds[run][tid][k], np.float64), 0, None)
            tis = p[:, 0] / max(p[:, 0].sum(), 1e-12)
            for source, chx in (("model", p[:, 1]), ("real", real_chx)):
                chx = chx / max(chx.sum(), 1e-12)
                for a in ALPHAS:
                    prof = fastloop.site_profiles(tis + a * chx, pos)
                    hits.setdefault((run, source, a), []).append(
                        pd.Series(fastloop.site_hits(prof), index=group.index)
                    )
    out = []
    for (run, source, a), parts in hits.items():
        h = pd.concat(parts).reindex(table.index)
        for grp, sel in [*table.groupby("group").groups.items(), ("nc_all", table.cls == "nc")]:
            mask = table.index.isin(sel) if not isinstance(sel, pd.Series) else sel.to_numpy()
            ref = table.rep1_hit[mask].mean()
            hit = h[mask].mean()
            out.append(
                {
                    "condition": c,
                    "run": run,
                    "chx": source,
                    "alpha": a,
                    "group": grp,
                    "n": int(mask.sum()),
                    "ref": ref,
                    "hit": hit,
                    "ratio": hit / ref,
                }
            )
    res = pd.DataFrame(out)
    res.to_csv(args.out, sep="\t", index=False, mode="a", header=not args.out.exists())
    summary = res.groupby(["chx", "alpha", "group"]).ratio.mean().unstack("group").round(3)
    print(c)
    print(summary.to_string())


if __name__ == "__main__":
    main()
