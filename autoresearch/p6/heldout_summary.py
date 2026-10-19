"""Combine the leave-one-cell-line-out scorecards into one held-out table.

Usage: python autoresearch/p6/heldout_summary.py <out.tsv> <out.md> <fold scorecard dir> [...]
For each condition, `rnacond_heldout` is the fold where that condition was held out;
`rnacond_seen` is the mean over folds where it was in training. Comparators (base, chx_winner,
rna_scaled) come from the held-out fold's scorecard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCALES = ["gene", "bin30", "codon", "nt", "frame"]
COLS = ["rnacond_heldout", "rnacond_seen", "base", "chx_winner", "rna_scaled"]


def main() -> None:
    out_tsv, out_md, dirs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(d) for d in sys.argv[3:]]
    tabs = [pd.read_csv(d / "scorecard.tsv", sep="\t") for d in dirs]
    allt = pd.concat(tabs, keys=range(len(tabs)), names=["fold"]).reset_index(level=0)
    key = ["condition", "library", "scale"]
    held = allt[allt.heldout.astype(str) == "True"]
    rows = []
    for (c, lib, sc), g in held.groupby(key):
        fold = g.fold.iloc[0]
        get = lambda src, t=g: t.loc[t.source == src]  # noqa: E731
        row = {"condition": c, "library": lib, "scale": sc, "yardstick": g.yardstick.iloc[0]}
        for src, name in (("rnacond", "rnacond_heldout"), ("base", "base"),
                          ("chx_winner", "chx_winner"), ("rna_scaled", "rna_scaled")):
            r = get(src)
            row[name] = r.value.iloc[0] if len(r) else float("nan")
            row[name + "_pct"] = r.percent.iloc[0] if len(r) else float("nan")
        seen = allt[(allt.condition == c) & (allt.library == lib) & (allt.scale == sc)
                    & (allt.source == "rnacond") & (allt.fold != fold)]
        row["rnacond_seen"] = seen.value.mean()
        row["rnacond_seen_pct"] = seen.percent.mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_tsv, sep="\t", index=False)
    lines = ["# RNA-conditioned model, leave-one-cell-line-out (val, chr7)", "",
             "Cells: Pearson r vs replicate 2 (percent of replicate 1 vs replicate 2). "
             "`rnacond_heldout`: model that never saw this cell line; `rnacond_seen`: mean of the "
             "folds that trained on it; `base`/`chx_winner`: experiment-1 per-condition models "
             "(trained on this line); `rna_scaled`: measured RNA x one scalar.", ""]
    fmt = lambda v, p: "–" if pd.isna(v) else (f"{v:.3f}" + (" (n/a)" if pd.isna(p) else f" ({p:.0f}%)"))  # noqa: E731
    for sc in SCALES:
        sub = df[df.scale == sc]
        if sub.empty:
            continue
        lines += [f"## {sc}", "", "| condition | library | rep1 vs rep2 | " + " | ".join(COLS) + " |",
                  "|" + "---|" * (3 + len(COLS))]
        for _, r in sub.sort_values(["library", "condition"]).iterrows():
            cells = [fmt(r[c], r[c + "_pct"]) for c in COLS]
            lines.append(f"| {r.condition} | {r.library} | {r.yardstick:.3f} | " + " | ".join(cells) + " |")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
