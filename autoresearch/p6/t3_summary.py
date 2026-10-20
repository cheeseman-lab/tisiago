"""Task-3 summary: K562 (held out) and seen-condition scorecard percents per arm vs reference.

Usage: python autoresearch/p6/t3_summary.py [scorecard dirs ...]
(default: every data/p6/scorecard/val_t3_* dir). The reference is the ``rnacond`` source
(ar_rnacond_k562_v2) read from the first dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCALES = ("gene", "bin30", "codon", "nt")


def rows_of(table: pd.DataFrame, source: str, name: str) -> dict:
    t = table[table.source == source]
    row = {"arm": name}
    for lib in ("tis", "chx"):
        k = t[(t.condition == "K562") & (t.library == lib)].set_index("scale").percent
        for scale in SCALES:
            row[f"K562_{lib}_{scale}"] = round(float(k.get(scale, float("nan"))), 1)
    seen = t[(t.condition != "K562") & (t.library == "tis")]
    for scale in ("codon", "nt"):
        row[f"seen_tis_{scale}"] = round(float(seen[seen.scale == scale].percent.mean()), 1)
    seen = t[(t.condition != "K562") & (t.library == "chx")]
    for scale in ("codon", "nt"):
        row[f"seen_chx_{scale}"] = round(float(seen[seen.scale == scale].percent.mean()), 1)
    frame = t[(t.condition == "K562") & (t.scale == "frame")].percent
    row["K562_frame"] = round(float(frame.iloc[0]), 1) if len(frame) else float("nan")
    return row


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or sorted(Path("data/p6/scorecard").glob("val_t3_*"))
    rows = []
    for i, d in enumerate(dirs):
        table = pd.read_csv(d / "scorecard.tsv", sep="\t")
        if i == 0:
            rows.append(rows_of(table, "rnacond", "reference_v2"))
        arm = [s for s in table.source.unique() if s not in ("rnacond", "rna_scaled")][0]
        rows.append(rows_of(table, arm, arm.removeprefix("ar_rnacond_k562_")))
    out = pd.DataFrame(rows).set_index("arm").T
    print(out.to_string())


if __name__ == "__main__":
    main()
