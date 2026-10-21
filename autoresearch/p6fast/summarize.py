"""Summarise autoresearch/p6fast/results.tsv: primary, K562 guards vs ref_rnacond_ps, seed groups.

Usage: python autoresearch/p6fast/summarize.py [--p2] [name-substring ...]
Seed replicates are grouped by stripping a trailing ``_s<k>`` from the name. ``--p2`` shows phase-2
rows only (``primary2`` scored), the per-start-class peak columns, and the phase-2 TIS guard.
"""

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REF = "ref_rnacond_ps"
GUARD_1 = ["K562_tis_gene", "K562_tis_bin30", "K562_chx_gene", "K562_chx_bin30"]
GUARD_3 = ["K562_chx_codon", "K562_chx_nt"]
# phase-2 guard: K562 TIS codon / nt not more than 2 below the phase-1 leader (r20_l300_chx25x2)
LEADER_TIS = {"K562_tis_codon": 76.4, "K562_tis_nt": 74.7}
SHOW2 = [
    "primary2",
    "primary",
    "K562_tis_codon",
    "K562_tis_nt",
    "K562_chx_bin30",
    "K562_chx_codon",
    "K562_chx_nt",
    "K562_chx_frame",
    "K562_tis_peak_meta_atg",
    "K562_tis_peak_meta_nc",
    "K562_tis_peak_argmax0_atg",
    "K562_tis_peak_argmax0_nc",
]
SHOW = [
    "primary",
    "K562_tis_codon",
    "K562_tis_nt",
    "K562_tis_bin30",
    "K562_chx_bin30",
    "K562_chx_codon",
    "K562_chx_nt",
    "K562_chx_frame",
    "K562_tis_peak_meta",
    "K562_tis_peak_argmax0",
]


def main(filters):
    p2 = "--p2" in filters
    filters = [f for f in filters if f != "--p2"]
    show = SHOW2 if p2 else SHOW
    df = pd.read_csv(HERE / "results.tsv", sep="\t")
    df = df.drop_duplicates("name", keep="last")
    ref = df[df.name == REF].iloc[0]
    if p2:
        df = df[df.primary2.notna()]
    fails = []
    for _, r in df.iterrows():
        bad = [c.replace("K562_", "") for c in GUARD_1 if r[c] < ref[c] - 1.0]
        bad += [c.replace("K562_", "") for c in GUARD_3 if r[c] < ref[c] - 3.0]
        if p2:
            bad += [c.replace("K562_", "") for c, v in LEADER_TIS.items() if r[c] < v - 2.0]
        fails.append(",".join(bad) or "pass")
    df["guards"] = fails
    df["group"] = [re.sub(r"_s\d+$", "", n) for n in df.name]
    if filters:
        df = df[df.name.str.contains("|".join(filters))]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(df[["name", *show, "guards"]].round(3).to_string(index=False))
    key = show[0]
    g = df.groupby("group")
    agg = g[show[: 12 if p2 else 7]].mean().round(3)
    agg[f"{key}_sd"] = g[key].std().round(2)
    agg["n"] = g.size()
    agg["guards_all"] = g["guards"].agg(lambda s: "pass" if (s == "pass").all() else "FAIL")
    print("\nseed groups (mean):")
    print(agg.sort_values(key, ascending=False).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
