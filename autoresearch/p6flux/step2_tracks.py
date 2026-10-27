"""Track metrics (record.json row, K562 val) of step-2 runs, per seed + mean.

Usage:
    python autoresearch/p6flux/step2_tracks.py [prefix ...]   # default sb sbctx sbflux
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def main() -> None:
    prefixes = sys.argv[1:] or ["sb", "sbctx", "sbflux"]
    rows = []
    for prefix in prefixes:
        for s in range(3):
            path = Path("data/p6/txtrain") / f"{prefix}_k562_s{s}" / "record.json"
            if not path.exists():
                continue
            row = json.loads(path.read_text())["row"]
            rows.append({"model": prefix, "seed": s, **{k: row.get(v) for k, v in COLS.items()}})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    g = df.drop(columns="seed").groupby("model")
    print()
    print(g.mean().round(3).to_string())
    print(g.std().round(3).to_string())


if __name__ == "__main__":
    main()
