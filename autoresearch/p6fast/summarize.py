"""Summarise autoresearch/p6fast/results.tsv: primary, K562 guards vs ref_rnacond_ps, seed groups.

Usage: python autoresearch/p6fast/summarize.py [name-substring ...]
Seed replicates are grouped by stripping a trailing ``_s<k>`` from the name.
"""

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REF = "ref_rnacond_ps"
GUARD_1 = ["K562_tis_gene", "K562_tis_bin30", "K562_chx_gene", "K562_chx_bin30"]
GUARD_3 = ["K562_chx_codon", "K562_chx_nt"]
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
    df = pd.read_csv(HERE / "results.tsv", sep="\t")
    df = df.drop_duplicates("name", keep="last")
    ref = df[df.name == REF].iloc[0]
    fails = []
    for _, r in df.iterrows():
        bad = [c.replace("K562_", "") for c in GUARD_1 if r[c] < ref[c] - 1.0]
        bad += [c.replace("K562_", "") for c in GUARD_3 if r[c] < ref[c] - 3.0]
        fails.append(",".join(bad) or "pass")
    df["guards"] = fails
    df["group"] = [re.sub(r"_s\d+$", "", n) for n in df.name]
    if filters:
        df = df[df.name.str.contains("|".join(filters))]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(df[["name", *SHOW, "guards"]].round(2).to_string(index=False))
    g = df.groupby("group")
    agg = g[SHOW[:7]].mean().round(2)
    agg["primary_sd"] = g["primary"].std().round(2)
    agg["n"] = g.size()
    agg["guards_all"] = g["guards"].agg(lambda s: "pass" if (s == "pass").all() else "FAIL")
    print("\nseed groups (mean):")
    print(agg.sort_values("primary", ascending=False).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
