"""Spec metrics of the flux round (autoresearch/p6flux/results.tsv), per config and seed-grouped."""

import re
import sys

import pandas as pd

COLS = {
    "primary3": "primary3",
    "tis_nt": "K562_tis_nt",
    "tis_codon": "K562_tis_codon",
    "chx_codon": "K562_chx_codon",
    "chx_nt": "K562_chx_nt",
    "chx_frame": "K562_chx_frame",
    "utr5_tis_ratio": "K562_utr5_tis_rep_ratio",
    "orf_frame": "K562_utr5_orf_frame",
    "orf_framed": "K562_utr5_orf_framed",
}

path = sys.argv[1] if len(sys.argv) > 1 else "autoresearch/p6flux/results.tsv"
df = pd.read_csv(path, sep="\t")
missing = [c for c in COLS.values() if c not in df.columns]
if missing:
    print("missing columns", missing)
table = df[["name", *[c for c in COLS.values() if c in df.columns]]].rename(
    columns={v: k for k, v in COLS.items()}
)
pd.set_option("display.width", 200)
print(table.round(3).to_string(index=False))
table["group"] = table.name.map(lambda n: re.sub(r"_s\d+$", "", n))
g = table.drop(columns="name").groupby("group")
out = g.mean().round(3)
out.insert(0, "n", g.size())
print()
print(out.to_string())
print()
print(g.std().round(3).to_string())
