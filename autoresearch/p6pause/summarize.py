"""Pause-pass readouts from a results TSV (fast loop or step-2 rescoring), K562 by default.

Groups rows by name with the seed suffix (``_s1``, ``_s2``) stripped and prints means (sd) of:
NC hit ratio in-CDS / out-of-CDS (all, first, later exon), ATG ratio, CHX pause ratio (and
pred / ref heights), utr5_orf_framed and the guards (primary3, TIS codon/nt, CHX codon/nt).

Usage:
    python autoresearch/p6pause/summarize.py [autoresearch/p6pause/results.tsv] [--condition K562]
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

COLS = {
    "nc_in": "tis_rep_ratio_nc_incds",
    "nc_in_first": "tis_rep_ratio_nc_first_incds",
    "nc_in_later": "tis_rep_ratio_nc_later_incds",
    "nc_out": "tis_rep_ratio_nc_outcds",
    "nc_out_first": "tis_rep_ratio_nc_first_outcds",
    "nc_out_later": "tis_rep_ratio_nc_later_outcds",
    "nc": "tis_rep_ratio_nc",
    "atg": "tis_rep_ratio_atg",
    "pause": "chx_pause_ratio",
    "pause_pred": "chx_pause_pred",
    "pause_ref": "chx_pause_ref",
    "framed": "utr5_orf_framed",
    "tis_codon": "tis_codon",
    "tis_nt": "tis_nt",
    "chx_codon": "chx_codon",
    "chx_nt": "chx_nt",
}
N_COLS = (
    "tis_rep_n_nc_first_incds",
    "tis_rep_n_nc_later_incds",
    "tis_rep_n_nc_outcds",
    "chx_pause_n",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="autoresearch/p6pause/results.tsv")
    ap.add_argument("--condition", default="K562")
    args = ap.parse_args()
    df = pd.read_csv(args.results, sep="\t")
    c = args.condition
    out = pd.DataFrame({"name": df["name"]})
    out["model"] = df["name"].map(lambda s: re.sub(r"_s\d+$", "", s))
    out["primary3"] = df["primary3"]
    for key, col in COLS.items():
        out[key] = df[f"{c}_{col}"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df[["name"] + [f"{c}_{n}" for n in N_COLS]].to_string(index=False))
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    grouped = out.drop(columns="name").groupby("model", sort=False)
    mean, sd, n = grouped.mean(), grouped.std(), grouped.size()
    print("\nmeans (n seeds)")
    print(mean.assign(n=n).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nsd")
    print(sd.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
