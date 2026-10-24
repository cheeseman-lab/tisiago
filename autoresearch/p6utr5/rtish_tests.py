"""Which Ribo-TISH test drops real 5'UTR starts on the predicted tracks (val K562).

Reads ``run_rtish_candidates.sh`` output (TIS and frame p-values, no thresholds, at every real
call, on predicted and real BAMs) and tabulates pass rates of the TIS test (p < .05), the frame
test (p < .05) and the full filter (both, Fisher q < .05) by TisType.

Usage:
    python autoresearch/p6utr5/rtish_tests.py --dir data/p6utr5/rtish_cand
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["Tid", "Start", "TisType", "TISCounts", "TISPvalue", "RiboPvalue", "FisherQvalue", "AALen"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/p6utr5/rtish_cand"))
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    real = pd.read_csv(args.dir / "real.txt", sep="\t", usecols=COLS)
    pred = pd.read_csv(args.dir / "predicted.txt", sep="\t", usecols=COLS)
    m = real.merge(pred, on=["Tid", "Start"], suffixes=("_r", "_p"))
    m["g"] = m.TisType_r.str.split(":").str[0]
    m = m[m.g.isin(["Annotated", "5'UTR", "Extended", "Truncated", "Novel"])]
    for side in ("r", "p"):
        m[f"tis_{side}"] = m[f"TISPvalue_{side}"] < 0.05
        m[f"frame_{side}"] = m[f"RiboPvalue_{side}"] < 0.05
        m[f"call_{side}"] = m[f"tis_{side}"] & m[f"frame_{side}"] & (m[f"FisherQvalue_{side}"] < 0.05)
    print(f"real calls re-tested: {len(m)} (Tid, start) rows")
    out = m.groupby("g").agg(
        n=("Start", "size"),
        call_real=("call_r", "mean"),
        call_pred=("call_p", "mean"),
        tis_real=("tis_r", "mean"),
        tis_pred=("tis_p", "mean"),
        frame_real=("frame_r", "mean"),
        frame_pred=("frame_p", "mean"),
        aalen=("AALen_r", "median"),
        tiscount_real=("TISCounts_r", "median"),
        tiscount_pred=("TISCounts_p", "median"),
    )
    print(out.round(3).to_string())
    print("\nfailure mode of real-called starts not called on predicted tracks (share):")
    miss = m[m.call_r & ~m.call_p]
    mode = np.select(
        [~miss.tis_p & ~miss.frame_p, ~miss.tis_p, ~miss.frame_p],
        ["both fail", "TIS fails", "frame fails"],
        "Fisher q only",
    )
    print(pd.crosstab(miss.g, mode, normalize="index").round(3).assign(n=miss.g.value_counts()))
    u = m[m.g == "5'UTR"]
    print("\n5'UTR by ORF length (aa):")
    print(
        u.groupby(pd.cut(u.AALen_r, [0, 10, 20, 40, 100, 1e6]), observed=True)
        .agg(n=("Start", "size"), call_pred=("call_p", "mean"), tis_pred=("tis_p", "mean"),
             frame_pred=("frame_p", "mean"), frame_real=("frame_r", "mean"))
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
