"""Tables for Q1-Q4 of the 5'UTR diagnosis from ``diagnose.py`` output (val K562 only).

Usage:
    python autoresearch/p6utr5/analyze.py --data data/p6utr5 > data/p6utr5/tables.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["Annotated", "5'UTR", "Extended", "Truncated", "Internal/Novel"]
MODELS = ["step2", "step1", "genomic"]


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n### {title}")
    print(df.round(3).to_string())


def q1(s: pd.DataFrame) -> None:
    for name, d in s.groupby("set"):
        d = d[d.real_win_frac.notna() & (d.real_win_frac > 0)]
        out = []
        for g in GROUPS:
            for cls, sel in (("all", True), ("ATG", d.atg), ("NC", ~d.atg)):
                x = d[(d.group == g) & sel]
                r2 = x[x.rep2_hit]
                rec = {
                    "group": g,
                    "codon": cls,
                    "n": len(x),
                    "real_hit": x.real_hit.mean(),
                    "both_reps": (x.rep1_hit & x.rep2_hit).mean(),
                    "n_rep2": len(r2),
                    "rep1|rep2": r2.rep1_hit.mean(),
                }
                for m in MODELS:
                    rec[f"{m}|rep2"] = r2[f"{m}_hit"].mean()
                    rec[f"{m}_ratio"] = r2[f"{m}_hit"].mean() / r2.rep1_hit.mean()
                rec["step2_all"] = x.step2_hit.mean()
                rec["rtish_pred_call"] = x.rtish_pred_call.mean()
                out.append(rec)
        show(f"Q1 reproducibility + model hit ({name} sites, real +-15 window > 0)", pd.DataFrame(out))


def q2(s: pd.DataFrame) -> None:
    d = s[(s.set == "cache") & (s.real_win_frac > 0)].copy()
    d["expr"] = pd.qcut(d.rna_per_nt.rank(method="first"), 3, labels=["low", "mid", "high"])
    cols = {}
    for src in ("real", "step2", "rep1"):
        for key in ("win_frac", "site_frac", "share3", "peakshare"):
            cols[f"{src}_{key}"] = (f"{src}_{key}", "median")
    agg = d.groupby(["group", "real_hit"]).agg(n=("tx_pos", "size"), **cols)
    show("Q2 signal at real starts (cache set; medians; fractions of transcript TIS mass)", agg.loc[GROUPS])
    d["ratio_win"] = d.step2_win_frac / d.real_win_frac
    d["ratio_site"] = d.step2_site_frac / d.real_site_frac.replace(0, np.nan)
    show(
        "Q2 step2/real mass ratio at site (median), reproducible starts only (both reps hit)",
        d[d.rep1_hit & d.rep2_hit]
        .groupby("group")
        .agg(n=("tx_pos", "size"), win=("ratio_win", "median"), site=("ratio_site", "median"))
        .loc[GROUPS],
    )
    # missed-site mode, reproducible starts only
    r = d[d.rep1_hit & d.rep2_hit & ~d.step2_hit].copy()
    absent = r.step2_win_frac < 0.25 * r.real_win_frac
    flat = ~absent & (r.step2_peakshare < 0.5 * r.real_peakshare)
    r["mode"] = np.where(absent, "absent", np.where(flat, "flat", "shifted"))
    show(
        "Q2 miss mode at reproducible missed starts (absent: window mass <1/4 real; flat: peak "
        "share <1/2 real; else shifted)",
        pd.crosstab(r.group, r["mode"], normalize="index").assign(n=r.group.value_counts()),
    )
    sh = r[r["mode"] == "shifted"]
    show(
        "Q2 shifted misses: model argmax offset (nt, 5'->3'), |off| quantiles + in-frame share",
        sh.groupby("group").agg(
            n=("tx_pos", "size"),
            med_abs=("step2_argoff", lambda x: np.median(np.abs(x))),
            downstream=("step2_argoff", lambda x: (x > 0).mean()),
            inframe=("step2_argoff", lambda x: (x % 3 == 0).mean()),
        ),
    )
    for g in ("5'UTR", "Annotated"):
        x = d[d.group == g]
        show(
            f"Q2 {g} by RNA expression tertile (all cache sites)",
            x.groupby("expr", observed=True).agg(
                n=("tx_pos", "size"),
                both_reps=("rep1_hit", lambda v: (v & x.loc[v.index, "rep2_hit"]).mean()),
                real_hit=("real_hit", "mean"),
                step2_hit=("step2_hit", "mean"),
                real_share3=("real_share3", "median"),
                step2_share3=("step2_share3", "median"),
                ratio_win=("ratio_win", "median"),
            ),
        )


def q3(s: pd.DataFrame, t: pd.DataFrame) -> None:
    d = s[(s.set == "cache") & (s.real_win_frac > 0) & (s.dist_cds > W_)].copy()
    d["log2_up_ann_real"] = np.log2((d.real_win_frac + 1e-4) / (d.real_annwin_frac + 1e-4))
    d["log2_up_ann_pred"] = np.log2((d.step2_win_frac + 1e-4) / (d.step2_annwin_frac + 1e-4))
    d["delta"] = d.log2_up_ann_pred - d.log2_up_ann_real
    d["same_block"] = np.where(
        (d.tx_pos // 1024 == d.cds_start // 1024) & ((d.tx_pos + 512) // 1024 == (d.cds_start + 512) // 1024),
        "both grids",
        np.where(
            (d.tx_pos // 1024 == d.cds_start // 1024) | ((d.tx_pos + 512) // 1024 == (d.cds_start + 512) // 1024),
            "one grid",
            "no",
        ),
    )
    d["dist_bin"] = pd.cut(d.dist_cds, [15, 32, 100, 300, 1000, 1e6])
    for g in ("5'UTR", "Extended"):
        x = d[d.group == g]
        show(
            f"Q3 {g}: log2(upstream-site +-15 mass / annotated +-15 mass), real vs step2 (median); "
            "delta<0 = model shifts mass to the annotated start",
            x.groupby("step2_hit").agg(
                n=("tx_pos", "size"),
                real=("log2_up_ann_real", "median"),
                pred=("log2_up_ann_pred", "median"),
                delta=("delta", "median"),
                ann_real=("real_annwin_frac", "median"),
                ann_pred=("step2_annwin_frac", "median"),
            ),
        )
        show(
            f"Q3 {g}: by distance to annotated start (nt)",
            x.groupby("dist_bin", observed=True).agg(
                n=("tx_pos", "size"), step2_hit=("step2_hit", "mean"), real_hit=("real_hit", "mean"),
                delta=("delta", "median"),
            ),
        )
        show(
            f"Q3 {g}: by 1 kb block sharing with annotated start",
            x.groupby("same_block").agg(
                n=("tx_pos", "size"), step2_hit=("step2_hit", "mean"), real_hit=("real_hit", "mean"),
                delta=("delta", "median"),
            ),
        )
    tt = t[(t.real_tis_total > 20) & (t.cds_start > 50)]
    out = {}
    for src in ("real", "rep1", "step2", "step1", "genomic"):
        out[src] = {
            "median utr5 frac": tt[f"{src}_utr5"].median(),
            "mean utr5 frac": tt[f"{src}_utr5"].mean(),
            "median ann frac": tt[f"{src}_ann"].median(),
        }
    show(f"Q3 transcript TIS mass fractions (n={len(tt)} tx, >20 real TIS reads, 5'UTR>50 nt)", pd.DataFrame(out).T)
    tt = tt.assign(r=np.log2((tt.step2_utr5 + 1e-4) / (tt.real_utr5 + 1e-4)),
                   r1=np.log2((tt.rep1_utr5 + 1e-4) / (tt.rep2_utr5 + 1e-4)))
    print("log2 step2/real 5'UTR fraction: median", round(tt.r.median(), 3),
          "| rep1/rep2:", round(tt.r1.median(), 3),
          "| corr(step2, real) of 5'UTR frac:", round(np.corrcoef(tt.step2_utr5, tt.real_utr5)[0, 1], 3),
          "| rep1 vs rep2:", round(np.corrcoef(tt.rep1_utr5, tt.rep2_utr5)[0, 1], 3))


def q4(s: pd.DataFrame, t: pd.DataFrame) -> None:
    d = s[(s.set == "cache") & (s.real_win_frac > 0) & (s.group == "5'UTR")].copy()
    rep = d[d.rep1_hit & d.rep2_hit]
    for name, x in (("all 5'UTR", d), ("reproducible 5'UTR", rep)):
        show(
            f"Q4 {name}: hit vs missed (step2)",
            x.groupby("step2_hit").agg(
                n=("tx_pos", "size"),
                atg=("atg", "mean"),
                cap_dist_med=("tx_pos", "median"),
                to_cds_med=("dist_cds", "median"),
                first_exon=("first_exon", "mean"),
                rna_per_nt=("rna_per_nt", "median"),
                upstream_of_rna_start=("tx_pos", lambda v: (v < x.loc[v.index, "rna_start"]).mean()),
                utr_rna_vs_cds=("rna_utr5_mean", lambda v: np.nanmedian(v / x.loc[v.index, "rna_cds_mean"])),
            ),
        )
        show(
            f"Q4 {name}: step2 hit by codon",
            x.assign(cod=np.where(x.atg, "ATG", np.where(x.codon.str.upper().isin(["CTG", "GTG", "TTG", "ACG"]), x.codon.str.upper(), "other NC")))
            .groupby("cod").agg(n=("tx_pos", "size"), step2=("step2_hit", "mean"), rep1=("rep1_hit", "mean"), real=("real_hit", "mean")),
        )
        show(
            f"Q4 {name}: step2 hit by distance from cap",
            x.groupby(pd.cut(x.tx_pos, [-1, 25, 50, 100, 200, 500, 1e6]), observed=True).agg(
                n=("tx_pos", "size"), step2=("step2_hit", "mean"), rep1=("rep1_hit", "mean"), atg=("atg", "mean")
            ),
        )
    tt = t[(t.cds_start > 50) & (t.rna_cds_mean > 0)]
    print("\nQ4 RNA-supported 5'UTR start (first 25-nt RNA mean >= 10% CDS mean), all K562 tx with 5'UTR>50:")
    print("  rna_start/MANE 5'UTR length median", round((tt.rna_start / tt.cds_start).median(), 3),
          "| fraction with rna_start > 50% of 5'UTR", round((tt.rna_start > 0.5 * tt.cds_start).mean(), 3),
          "| median RNA 5'UTR mean / CDS mean", round((tt.rna_utr5_mean / tt.rna_cds_mean).median(), 3))


W_ = 15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/p6utr5"))
    args = ap.parse_args()
    s = pd.read_parquet(args.data / "sites.parquet")
    t = pd.read_parquet(args.data / "transcripts.parquet")
    pd.set_option("display.width", 250)
    q1(s)
    q2(s)
    q3(s, t)
    q4(s, t)


if __name__ == "__main__":
    main()
