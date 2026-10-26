"""Ribo-TISH start-site agreement (K562 val, predicted vs real calls) for step-2 runs, per seed + mean.

Usage:
    python autoresearch/p6flux/compare_ribotish.py [prefix ...]   # default sb sbctx sbflux
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("data/p6/txtrain/export")
REPORT = "predicted_val_floatresample_s0/calls/K562/report.json"
TYPES = ("5'UTR", "Annotated", "Extended", "Truncated")


def main() -> None:
    prefixes = sys.argv[1:] or ["sb", "sbctx", "sbflux"]
    rows = []
    for prefix in prefixes:
        for s in range(3):
            path = ROOT / f"{prefix}_k562_s{s}" / REPORT
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            row = {"model": prefix, "seed": s, "J": r["overall"]["jaccard"]}
            row["R"] = r["overall"]["recall"]
            row["P"] = r["overall"]["precision"]
            row["J_nc"] = r["by_start_codon"]["near_cognate"]["jaccard"]
            for t in TYPES:
                v = r["by_tis_type"].get(t, {})
                row[f"R_{t}"] = v.get("recall")
                row[f"P_{t}"] = v.get("precision")
            rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    g = df.drop(columns="seed").groupby("model")
    print()
    print(g.mean().round(3).to_string())
    print(g.std().round(3).to_string())


if __name__ == "__main__":
    main()
