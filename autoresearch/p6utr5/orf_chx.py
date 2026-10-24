"""CHX (elongation) signal inside the ORF of each real start: real vs step-2 predicted (val K562).

Ribo-TISH keeps a start only if its ORF passes the frame test on the CHX track (in-frame >
out-of-frame P-sites from start to stop). For each site of ``diagnose.py``'s ``sites.parquet``
this reads the ORF (first in-frame stop in the MANE mRNA) and reports, for real pooled CHX and the
step-2 CHX export: in-frame fraction, CHX per nt in the ORF, and that density relative to the CDS.

Usage:
    python autoresearch/p6utr5/orf_chx.py --data data/p6utr5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func.predict_tracks import strand_shift
from tisiago.seq2func.scorecard import BigWigs
from tisiago.seq2func.txcache import TxCache, spliced_values, unit_of

STOPS = {(3, 0, 0), (3, 0, 2), (3, 2, 0)}  # TAA TAG TGA in A C G T = 0..3


def orf_stop(seq: np.ndarray, start: int) -> int:
    for i in range(start, len(seq) - 2, 3):
        if (int(seq[i]), int(seq[i + 1]), int(seq[i + 2])) in STOPS:
            return i + 3
    return len(seq) - (len(seq) - start) % 3


def frame_stats(x: np.ndarray, start: int, stop: int) -> tuple[float, float]:
    orf = x[start:stop]
    tot = orf.sum()
    return (orf[0::3].sum() / tot if tot > 0 else np.nan), tot / max(stop - start, 1)


def cds_frame_share(x: np.ndarray, start: int, stop: int, cds_start: int) -> float:
    orf = x[start:stop]
    tot = orf.sum()
    phase = (cds_start - start) % 3  # ORF offsets in the annotated CDS frame
    return orf[phase::3].sum() / tot if tot > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/p6utr5"))
    ap.add_argument("--cache", type=Path, default=Path("data/p6/txcache/fc2d_k562"))
    ap.add_argument("--step2", type=Path, default=Path("data/p6/txtrain/export/sb_k562_s0/K562"))
    args = ap.parse_args()
    cache = TxCache(args.cache)
    k = cache.condition_index("K562")
    val = {r.transcript_id: r for r in cache.rows("val").itertuples()}
    sites = pd.read_parquet(args.data / "sites.parquet")
    chx = cache.array("val_chr7", "chx")
    seq = cache.array("val_chr7", "seq")
    reader = BigWigs()
    out = []
    for tid, grp in sites.groupby("transcript_id"):
        r = val[tid]
        lo, hi = int(r.offset), int(r.offset) + int(r.length)
        real = np.asarray(chx[k, lo:hi, 2], np.float64)
        unit = unit_of(r)
        f = args.step2 / f"chx_pooled.{'plus' if unit.strand == '+' else 'minus'}.bw"
        pred = spliced_values(reader, f, unit, strand_shift(unit.strand))
        s = np.asarray(seq[lo:hi])
        cs, ce = int(r.cds_start), int(r.cds_end)
        cds_real = real[cs:ce].mean() if ce > cs else np.nan
        cds_pred = pred[cs:ce].mean() if ce > cs else np.nan
        for row in grp.itertuples():
            stop = orf_stop(s, row.tx_pos)
            fr, dr = frame_stats(real, row.tx_pos, stop)
            fp, dp = frame_stats(pred, row.tx_pos, stop)
            out.append(
                {
                    "set": row.set,
                    "transcript_id": tid,
                    "tx_pos": row.tx_pos,
                    "orf_len": stop - row.tx_pos,
                    "chx_frame_real": fr,
                    "chx_frame_pred": fp,
                    "chx_dens_real": dr,
                    "chx_dens_pred": dp,
                    "chx_rel_real": dr / cds_real if cds_real > 0 else np.nan,
                    "chx_rel_pred": dp / cds_pred if cds_pred > 0 else np.nan,
                    "chx_cdsframe_real": cds_frame_share(real, row.tx_pos, stop, cs),
                    "chx_cdsframe_pred": cds_frame_share(pred, row.tx_pos, stop, cs),
                }
            )
    reader.close()
    orf = pd.DataFrame(out)
    sites = sites.drop(columns=[c for c in orf.columns if c.startswith(("chx_", "orf_"))], errors="ignore")
    sites.merge(orf, on=["set", "transcript_id", "tx_pos"]).to_parquet(args.data / "sites_orf.parquet")
    print(f"{len(orf)} ORFs -> {args.data / 'sites_orf.parquet'}")


if __name__ == "__main__":
    main()
