"""Why the flux CHX stays weakly framed at 5'UTR starts: frame purity of the TIS that feeds it.

For cache 5'UTR starts (data/p6utr5/sites.parquet, K562 val) with >= 10 real CHX reads on the ORF:
TIS frame purity (in-frame share of TIS mass within +-15 nt of the start) and the in-frame share of
a flux recurrence (keep 0.978 per codon, no stops, as learned by f3_replace) over the ORF, driven by
the real TIS, the base head's TIS (step 1, ts_k562_s0 export) or the flux head's TIS
(f3_replace_s1),
next to the model's own CHX in-frame share. ``fluxstop`` is the same recurrence terminated at
stop codons (diagnostic ceiling for a stop hazard that has learned stop codons; not a model input).

Usage:
    python autoresearch/p6flux/tis_frame.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func import fastloop
from tisiago.seq2func.predict_tracks import strand_shift
from tisiago.seq2func.scorecard import BigWigs
from tisiago.seq2func.txcache import TxCache, spliced_values, unit_of

EXPORTS = {
    "base": Path("data/p6/fastloop/export_twostage/ts_k562_s0/K562"),
    "flux": Path("data/p6/fastloop/export_flux/f3_replace_s1/K562"),
}
KEEP = 0.978


def flux_of(tis: np.ndarray, stops: np.ndarray | None = None) -> np.ndarray:
    """Per-frame flux; ``stops`` (bool per nt) terminates all flux at that codon."""
    out = np.zeros_like(tis)
    for f in range(3):
        acc = 0.0
        for i in range(f, len(tis), 3):
            acc = acc * KEEP * (0.0 if stops is not None and stops[i] else 1.0) + tis[i]
            out[i] = acc
    return out


def stop_positions(seq: np.ndarray) -> np.ndarray:
    codes = np.zeros(len(seq), bool)
    for i in range(len(seq) - 2):
        codes[i] = tuple(int(x) for x in seq[i : i + 3]) in fastloop.STOP_CODES
    return codes


def frame_share(x: np.ndarray, start: int, lo: int, hi: int) -> float:
    seg = x[lo:hi]
    total = seg.sum()
    idx = np.arange(lo, hi)
    return float(seg[(idx - start) % 3 == 0].sum() / total) if total > 0 else np.nan


def main() -> None:
    cache = TxCache("data/p6/txcache/fc2d_k562")
    k = cache.condition_index("K562")
    val = {r.transcript_id: r for r in cache.rows("val").itertuples()}
    sites = pd.read_parquet("data/p6utr5/sites.parquet")
    sites = sites[(sites.set == "cache") & (sites.tistype == "5'UTR")]
    sites = sites.drop_duplicates(["transcript_id", "tx_pos"])
    tis_arr, chx_arr = cache.array("val_chr7", "tis"), cache.array("val_chr7", "chx")
    seq = cache.array("val_chr7", "seq")
    reader = BigWigs()
    rows = []
    for tid, grp in sites.groupby("transcript_id"):
        r = val[tid]
        lo, hi = int(r.offset), int(r.offset) + int(r.length)
        unit = unit_of(r)
        sfx = "plus" if unit.strand == "+" else "minus"
        tracks = {
            "real": (
                np.asarray(tis_arr[k, lo:hi, 2], float),
                np.asarray(chx_arr[k, lo:hi, 2], float),
            )
        }
        for name, path in EXPORTS.items():
            tracks[name] = tuple(
                np.asarray(
                    spliced_values(
                        reader, path / f"{lib}_pooled.{sfx}.bw", unit, strand_shift(unit.strand)
                    ),
                    float,
                )
                for lib in ("tis", "chx")
            )
        s = np.asarray(seq[lo:hi])
        fluxes = {n: flux_of(t) for n, (t, _) in tracks.items()}
        stops = stop_positions(s)
        fluxes_stop = {n: flux_of(t, stops) for n, (t, _) in tracks.items()}
        for site in grp.itertuples():
            start = int(site.tx_pos)
            stop = fastloop.orf_stop(s, start)
            if tracks["real"][1][start:stop].sum() < fastloop.UTR5_MIN_ORF_READS:
                continue
            row = {"transcript_id": tid, "tx_pos": start, "orf_nt": stop - start}
            w0, w1 = max(0, start - 15), min(len(s), start + 16)
            for n, (t, c) in tracks.items():
                row[f"tis_purity_{n}"] = frame_share(t, start, w0, w1)
                row[f"flux_frame_{n}"] = frame_share(fluxes[n], start, start, stop)
                row[f"fluxstop_frame_{n}"] = frame_share(fluxes_stop[n], start, start, stop)
                row[f"chx_frame_{n}"] = frame_share(c, start, start, stop)
            rows.append(row)
    reader.close()
    df = pd.DataFrame(rows)
    out = Path("data/p6flux")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "tis_frame.parquet")
    prefixes = ("tis_purity", "flux_frame", "fluxstop_frame", "chx_frame")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    print(f"n = {len(df)} 5'UTR ORFs")
    summary = pd.DataFrame({"mean": df[cols].mean(), "framed(>0.5)": (df[cols] > 0.5).mean()})
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
