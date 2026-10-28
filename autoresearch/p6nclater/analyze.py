"""Splits of the later-exon near-cognate miss on reproducible starts (tables from ``sites.py``).

A start is reproducible when replicate 2's TIS argmax over +-30 nt is within 1 nt of it. On those
starts: ``ref`` = P(replicate-1 hit), ``hit`` = P(model hit) (mean of the runs' float
predictions), ``ratio`` = hit / ref. The later-exon NC gap is ``sum(ref - hit)`` over later-exon
NC starts (shortfall vs replicate). A binary factor F "explains" ``n_F * ref_F * (ratio_notF -
ratio_F) / gap`` of it (the hits recovered if F starts did as well as the rest).

Usage:
    python autoresearch/p6nclater/analyze.py --data data/p6nclater \
        > autoresearch/p6nclater/report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def stats(d: pd.DataFrame) -> pd.Series:
    n = len(d)
    ref = d.rep1_hit.mean() if n else np.nan
    hit = d.model_hit.mean() if n else np.nan
    return pd.Series(
        {
            "n": n,
            "ref": ref,
            "hit": hit,
            "ratio": hit / ref if n and ref > 0 else np.nan,
            "shortfall": float((d.rep1_hit - d.model_hit).sum()),
            "rel": d.model_rel.mean() if n else np.nan,
        }
    )


def split_table(d: pd.DataFrame, col: str) -> pd.DataFrame:
    t = d.groupby(col, observed=True).apply(stats, include_groups=False)
    t["share_n"] = t.n / t.n.sum()
    total = t.shortfall.sum()
    t["share_gap"] = t.shortfall / total if total else np.nan
    return t


def explained(d: pd.DataFrame, flag: pd.Series) -> dict:
    """Share of ``d``'s shortfall recovered if flagged starts had the unflagged ratio."""
    f, nf = d[flag], d[~flag]
    gap = float((d.rep1_hit - d.model_hit).sum())
    if not len(f) or not len(nf) or gap <= 0:
        return {"n_F": len(f), "explained": np.nan}
    sf, snf = stats(f), stats(nf)
    rec = sf.n * sf.ref * (snf.ratio - sf.ratio)
    return {
        "n_F": int(sf.n),
        "frac_F": len(f) / len(d),
        "ratio_F": sf.ratio,
        "ratio_notF": snf.ratio,
        "explained": rec / gap,
    }


def tis_group(t: str) -> str:
    t = str(t).split(":")[0]
    return {"Novel": "Novel/Internal", "Internal": "Novel/Internal"}.get(t, t)


def add_features(d: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for thr in (100, 200, 500):
        d[f"alt5_{thr}"] = (d.d_full <= thr) & (d.d_full < d.tx_pos - 50)
    d["alt5_basic_200"] = (d.d_basic <= 200) & (d.d_basic < d.tx_pos - 50)
    d["alt5_any_200"] = (d.d_all <= 200) & (d.d_all < d.tx_pos - 50)
    d["alt_first_exon"] = d.first_full & (d.region == "later")
    d["alt_gain"] = np.log10((d.tx_pos + 1) / (d.d_full.fillna(d.tx_pos) + 1))
    q = controls.rna_step.quantile([1 / 3, 2 / 3]).to_numpy()
    d["rna_step_bin"] = pd.cut(
        d.rna_step, [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"]
    )
    qp = controls.rna_peak_rel.quantile(0.9)
    d["rna_peak_top10"] = d.rna_peak_rel > qp
    d["rna_alt_tss"] = d["alt5_200"] & (d.rna_step > q[1])
    d["expr_bin"] = pd.qcut(d.tx_tis_reads, 3, labels=["low", "mid", "high"])
    d["junc_bin"] = pd.cut(
        d.dist_junction_up, [-2, 0, 50, 200, np.inf], labels=["first", "<=50", "50-200", ">200"]
    )
    d["nj_bin"] = pd.cut(d.n_junctions_up, [-1, 0, 1, 3, np.inf], labels=["0", "1", "2-3", "4+"])
    d["pos5_bin"] = pd.cut(
        d.tx_pos, [-1, 200, 500, 1000, np.inf], labels=["<200", "200-500", "500-1k", ">1k"]
    )
    d["codon_grp"] = np.where(d.codon.isin(["CTG", "GTG", "TTG", "ACG"]), d.codon, "other")
    d["tis_grp"] = d.tistype.map(tis_group)
    d["deep_cds"] = d.frame.str.startswith("cds") & (d.dist_cds_start > 0)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/p6nclater"))
    args = ap.parse_args()
    frames = [pd.read_parquet(p) for p in sorted(args.data.glob("sites_*.parquet"))]
    allsites = pd.concat(frames, ignore_index=True)
    for line, tab in allsites.groupby("condition"):
        real = tab[tab.real]
        bad = (real.d_all > real.tx_pos).mean()
        print(f"[{line}] starts {len(real)}, controls {(~tab.real).sum()}; d_all>tx_pos {bad:.3f}")
    print()
    for line, tab in [*allsites.groupby("condition"), ("both", allsites)]:
        controls = tab[~tab.real]
        tab = add_features(tab, controls)
        controls = tab[~tab.real]
        rep = tab[tab.real & tab.rep2_hit]
        print("=" * 100)
        print(f"{line}: reproducible starts by class x region")
        print(rep.groupby(["cls", "region"]).apply(stats, include_groups=False).round(3))
        nc = rep[rep.cls == "nc"]
        later = nc[nc.region == "later"]
        first = nc[nc.region == "first"]
        sl, sf = stats(later), stats(first)
        gap_vs_first = sl.n * sl.ref * (sf.ratio - sl.ratio)
        print(
            f"\nlater NC shortfall vs replicate {sl.shortfall:.1f} hits of {sl.n:.0f};"
            f" vs first-exon ratio {gap_vs_first:.1f}"
        )
        # Q1/Q2 enrichment: alt 5' ends near the start, vs controls
        print("\nfeature prevalence (later exon): repNC / all NC starts / NC controls / rep ATG")
        ctl_l = controls[controls.region == "later"]
        nc_all_l = tab[tab.real & (tab.cls == "nc") & (tab.region == "later")]
        atg_l = rep[(rep.cls == "atg") & (rep.region == "later")]
        rows = {}
        for col in (
            "alt5_100",
            "alt5_200",
            "alt5_500",
            "alt5_basic_200",
            "alt5_any_200",
            "alt_first_exon",
            "rna_peak_top10",
            "rna_alt_tss",
        ):
            rows[col] = [
                later[col].mean(),
                nc_all_l[col].mean(),
                ctl_l[col].mean(),
                atg_l[col].mean(),
            ]
        rows["rna_step_median"] = [
            later.rna_step.median(),
            nc_all_l.rna_step.median(),
            ctl_l.rna_step.median(),
            atg_l.rna_step.median(),
        ]
        rows["d_full_median"] = [
            later.d_full.median(),
            nc_all_l.d_full.median(),
            ctl_l.d_full.median(),
            atg_l.d_full.median(),
        ]
        rows["tx_pos_median"] = [
            later.tx_pos.median(),
            nc_all_l.tx_pos.median(),
            ctl_l.tx_pos.median(),
            atg_l.tx_pos.median(),
        ]
        rows["n"] = [len(later), len(nc_all_l), len(ctl_l), len(atg_l)]
        print(pd.DataFrame(rows, index=["repNC", "allNC", "ctlNC", "repATG"]).T.round(3))
        # Q3/Q4 splits of the later-exon NC hit rate
        print("\nlater-exon rep NC: binary factors (explained = share of shortfall vs replicate)")
        ex = {}
        for col in (
            "alt5_100",
            "alt5_200",
            "alt5_500",
            "alt5_basic_200",
            "alt5_any_200",
            "alt_first_exon",
            "rna_peak_top10",
            "rna_alt_tss",
            "deep_cds",
        ):
            ex[col] = explained(later, later[col].astype(bool))
        print(pd.DataFrame(ex).T.round(3))
        for col in (
            "rna_step_bin",
            "expr_bin",
            "junc_bin",
            "nj_bin",
            "pos5_bin",
            "codon_grp",
            "frame",
            "tis_grp",
        ):
            print(f"\nlater NC by {col}")
            print(split_table(later, col).round(3))
        print("\nfirst-exon reproducible NC by pos5_bin / frame / tis_grp")
        for col in ("pos5_bin", "frame", "tis_grp", "expr_bin"):
            print(split_table(first, col).round(3))
        print("\nall reproducible NC: region x pos5_bin")
        print(
            nc.groupby(["pos5_bin", "region"], observed=True)
            .apply(stats, include_groups=False)
            .round(3)
        )
        print("\nall reproducible NC: region x deep_cds")
        print(
            nc.groupby(["deep_cds", "region"], observed=True)
            .apply(stats, include_groups=False)
            .round(3)
        )
        print("\nall reproducible NC: region x alt5_200")
        print(
            nc.groupby(["alt5_200", "region"], observed=True)
            .apply(stats, include_groups=False)
            .round(3)
        )
        decompose(nc, tab)


def decompose(nc: pd.DataFrame, tab: pd.DataFrame) -> None:
    """Exclusive start categories, their shortfall shares and the composition effect."""
    deep = nc.deep_cds
    nc = nc.assign(
        grp=np.select(
            [
                ~deep,
                deep & (nc.dist_cds_start <= 30),
                deep & (nc.dist_cds_start <= 100),
                deep & (nc.d_full < nc.dist_cds_start),
            ],
            ["A_not_in_cds", "B_cds<=30", "C_cds30-100", "D_cds>100_alt5_past_aug"],
            "E_cds>100_no_alt5",
        ),
        chx_share=nc.chx_site / nc.chx_win.replace(0, np.nan),
    )
    print("\nexclusive categories (A upstream/extension/uORF; B-E in-frame inside MANE CDS)")
    for region in ("first", "later"):
        t = nc[nc.region == region].groupby("grp").apply(stats, include_groups=False)
        t["share_gap"] = t.shortfall / t.shortfall.sum()
        extra = nc[nc.region == region].groupby("grp")[["tis_chx_enrich", "chx_share"]].median()
        print(region)
        print(t.join(extra).round(3))
    first, later = nc[nc.region == "first"], nc[nc.region == "later"]
    by = {key: stats(g) for key, g in first.groupby("deep_cds")}
    pred = sum(by[key].ratio * g.rep1_hit.sum() for key, g in later.groupby("deep_cds"))
    print(
        f"later ratio {stats(later).ratio:.3f}, first {stats(first).ratio:.3f}; later predicted"
        f" from first-exon in/out-of-CDS ratios {pred / later.rep1_hit.sum():.3f};"
        f" in-CDS share first {first.deep_cds.mean():.2f} later {later.deep_cds.mean():.2f}"
    )
    ctl = tab[~tab.real & (tab.frame == "cds_inframe") & (tab.dist_cds_start > 100)]
    rep = nc[nc.deep_cds & (nc.dist_cds_start > 100)]
    freq = pd.DataFrame(
        {
            "rep_cds>100": rep.codon.value_counts(normalize=True),
            "ctl_cds_inframe": ctl.codon.value_counts(normalize=True),
        }
    ).fillna(0)
    freq["enrichment"] = freq.iloc[:, 0] / freq.iloc[:, 1]
    print("codons of in-CDS (>100 nt) reproducible NC vs in-frame CDS NC codons")
    print(freq.round(3).T)
    print(
        f"TIS/CHX log2 enrichment median: in-CDS>100 NC {rep.tis_chx_enrich.median():.2f},"
        f" in-frame CDS controls {ctl.tis_chx_enrich.median():.2f}"
    )


if __name__ == "__main__":
    main()
