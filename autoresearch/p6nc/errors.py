"""Q2 error analysis and Q3 replicate ceiling at real K562 starts (val chr7, final model).

Usage:
    python autoresearch/p6nc/errors.py --data data/p6nc --out data/p6nc/errors

Sites: K562 pooled Ribo-TISH starts on K562-expressed MANE val transcripts with real pooled TIS
signal within +-30 nt (as ``autoresearch/p6/tis_peak_share.py``). A source "hits" a site when its
argmax over +-30 nt is within 1 nt of the start (``within1``; ``argmax0`` = exactly at it).
Missed = predicted argmax not within 1 nt. Offsets are 5'->3' (positive = downstream).
Writes ``sites.parquet`` and TSV tables under ``--out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FLANK = 30
WIDE = 90


def tis_group(t: str) -> str:
    t = str(t)
    if t.startswith("Annotated"):
        return "Annotated"
    if t.startswith("Extended"):
        return "Extended"
    if t.startswith("Truncated"):
        return "Truncated"
    if t.startswith("5'UTR"):
        return "uORF(5'UTR)"
    if t.startswith("3'UTR"):
        return "3'UTR"
    return "Novel/Internal"


def summarize(d: pd.DataFrame) -> dict:
    out = {
        "n": len(d),
        "real_within1": d.real_hit.mean(),
        "rep1_within1": d.rep1_hit.mean(),
        "rep2_within1": d.rep2_hit.mean(),
        "both_reps_within1": d.both_hit.mean(),
        "pred_within1": d.pred_hit.mean(),
        "pred_argmax0": d.pred_argmax0.mean(),
        "pred_within1|both_reps": d.loc[d.both_hit, "pred_hit"].mean() if d.both_hit.any() else np.nan,
        "real_share0": d.real_share0.mean(),
        "pred_share0": d.pred_share0.mean(),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/p6nc"))
    ap.add_argument("--out", type=Path, default=Path("data/p6nc/errors"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t = pd.read_parquet(args.data / "codons.parquet")
    tx_mean = t.groupby("transcript_id").rna_tx_mean.first()
    dec = pd.qcut(tx_mean.rank(method="first"), 10, labels=False)
    t["decile"] = t.transcript_id.map(dec).astype(int)
    prof = np.load(args.data / "profiles.npz")
    P, site_rows = prof["profiles"], prof["site_rows"]  # (n, 5, 181): rep1 rep2 pooled pred ckpt
    peaks = pd.read_parquet(args.data / "pred_peaks.parquet")
    s = t.iloc[site_rows].copy().reset_index(names="row")
    c = WIDE
    win = slice(c - FLANK, c + FLANK + 1)
    offs = np.arange(-FLANK, FLANK + 1)

    def arg(k):
        w = P[:, k, win]
        return offs[w.argmax(1)], w.sum(1)

    for k, name in enumerate(("rep1", "rep2", "real", "pred", "ckpt")):
        o, tot = arg(k)
        s[f"{name}_o"] = np.where(tot > 0, o, 999)
        s[f"{name}_tot"] = tot
        s[f"{name}_share0"] = np.where(tot > 0, P[:, k, c] / np.maximum(tot, 1e-12), 0.0)
        s[f"{name}_ratio"] = P[:, k, c] / np.maximum(P[:, k, win].max(1), 1e-12)
    s = s[s.real_tot > 0].copy()
    s["cls"] = np.where(s.codon == "ATG", "ATG", "NC")
    for name in ("rep1", "rep2", "real", "pred", "ckpt"):
        s[f"{name}_hit"] = s[f"{name}_o"].abs() <= 1
    s["both_hit"] = s.rep1_hit & s.rep2_hit
    s["pred_argmax0"] = s.pred_o == 0
    s["group"] = s.tistype.map(tis_group)
    s["m3"] = s.ctx43.str[17]
    s["p4"] = s.ctx43.str[23]
    s["kozak"] = np.select(
        [s.m3.isin(["A", "G"]) & (s.p4 == "G"), s.m3.isin(["A", "G"]) | (s.p4 == "G")],
        ["strong", "adequate"],
        "weak",
    )
    s["dist5"] = pd.cut(s.tx_pos, [-1, 50, 150, 500, 1e9], labels=["<=50", "51-150", "151-500",
                                                                   ">500"])
    # what sits at the predicted argmax (missed sites)
    cod_by = {k: g.set_index("tx_pos") for k, g in t.groupby("transcript_id")}
    call_pos = {k: set(g.tx_pos[g.label]) for k, g in t.groupby("transcript_id")}
    pk_by = {k: g.tx_pos.to_numpy() for k, g in peaks.groupby("transcript_id")}
    at_atg, at_call, at_cds, same_frame, real_agree = [], [], [], [], []
    near_d, near_atg, near_cds, near_call, rel_ratio, real_ratio = [], [], [], [], [], []
    for r in s.itertuples():
        q = r.tx_pos + r.pred_o
        g = cod_by[r.transcript_id]
        at_atg.append(q in g.index and g.at[q, "codon"] == "ATG")
        at_call.append(q in call_pos[r.transcript_id] and q != r.tx_pos)
        at_cds.append(q == r.cds_start)
        same_frame.append(r.pred_o % 3 == 0)
        real_agree.append(abs(r.pred_o - r.real_o) <= 1)
        pk = pk_by.get(r.transcript_id, np.array([], int))
        pk = pk[np.abs(pk - r.tx_pos) > 1]
        if len(pk):
            d = pk[np.argmin(np.abs(pk - r.tx_pos))] - r.tx_pos
            p = r.tx_pos + d
            near_d.append(int(d))
            near_atg.append(p in g.index and g.at[p, "codon"] == "ATG")
            near_cds.append(p == r.cds_start)
            near_call.append(p in call_pos[r.transcript_id])
        else:
            near_d.append(np.nan)
            near_atg.append(False)
            near_cds.append(False)
            near_call.append(False)
    s["pred_at_atg"] = at_atg
    s["pred_at_other_call"] = at_call
    s["pred_at_cds_start"] = at_cds
    s["pred_same_frame"] = same_frame
    s["pred_agrees_real_argmax"] = real_agree
    s["nearest_peak_d"] = near_d
    s["nearest_peak_atg"] = near_atg
    s["nearest_peak_cds"] = near_cds
    s["nearest_peak_call"] = near_call
    s.to_parquet(args.out / "sites.parquet", index=False)

    tables = {}
    tables["q3_by_class"] = pd.DataFrame(
        {cls: summarize(g) for cls, g in s.groupby("cls")}
    ).T
    nc = s[s.cls == "NC"]
    atg = s[s.cls == "ATG"]
    for col in ("codon", "group", "region", "kozak", "decile", "dist5"):
        tables[f"nc_by_{col}"] = pd.DataFrame(
            {str(k): summarize(g) for k, g in nc.groupby(col, observed=True)}
        ).T
    tables["atg_by_group"] = pd.DataFrame(
        {str(k): summarize(g) for k, g in atg.groupby("group", observed=True)}
    ).T

    def miss_table(d: pd.DataFrame) -> pd.Series:
        m = d[~d.pred_hit]
        up = (m.pred_o < -1).mean()
        return pd.Series(
            {
                "missed": len(m),
                "missed_frac": len(m) / max(len(d), 1),
                "pred_argmax_upstream": up,
                "pred_argmax_downstream": (m.pred_o > 1).mean(),
                "median_|offset|": m.pred_o.abs().median(),
                "pred_argmax_same_frame": m.pred_same_frame.mean(),
                "pred_argmax_at_ATG": m.pred_at_atg.mean(),
                "pred_argmax_at_other_K562_call": m.pred_at_other_call.mean(),
                "pred_argmax_at_annotated_start": m.pred_at_cds_start.mean(),
                "pred_argmax=real_argmax(+-1)": m.pred_agrees_real_argmax.mean(),
                "real_pooled_also_offsite": (~m.real_hit).mean(),
                "reproducible(both reps at start)": m.both_hit.mean(),
                "nearest_pred_peak_|d|_median": m.nearest_peak_d.abs().median(),
                "nearest_pred_peak_downstream": (m.nearest_peak_d > 0).mean(),
                "nearest_pred_peak_at_ATG": m.nearest_peak_atg.mean(),
                "nearest_pred_peak_at_annotated": m.nearest_peak_cds.mean(),
                "nearest_pred_peak_at_K562_call": m.nearest_peak_call.mean(),
                "pred_share0_missed": m.pred_share0.mean(),
                "real_share0_missed": m.real_share0.mean(),
                "pred_start/localmax_median": m.pred_ratio.median(),
                "real_start/localmax_median": m.real_ratio.median(),
            }
        )

    tables["q2_missed"] = pd.DataFrame(
        {
            "NC all": miss_table(nc),
            "NC reproducible": miss_table(nc[nc.both_hit]),
            "ATG all": miss_table(atg),
            "ATG reproducible": miss_table(atg[atg.both_hit]),
        }
    )
    # missed reproducible NC: offset histogram
    m = nc[nc.both_hit & ~nc.pred_hit]
    bins = [-31, -10, -4, -2, 1, 3, 9, 30]
    tables["q2_offset_hist_nc_reproducible_missed"] = (
        pd.cut(m.pred_o, bins).value_counts(sort=False).to_frame("n")
    )
    for name, tab in tables.items():
        tab.to_csv(args.out / f"{name}.tsv", sep="\t")
        with pd.option_context("display.width", 220, "display.precision", 3,
                               "display.max_columns", 30):
            print(f"\n== {name}\n{tab}")


if __name__ == "__main__":
    main()
