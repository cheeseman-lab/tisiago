"""Assemble the Phase-3/4 head-comparison table from saved dense-TEST prediction .npz files.

Each scoring run (dense_caller --save-preds, efficiency_head --save-preds) writes y/cognate/tx and
one or more ``p::<head>`` arrays. This recomputes the headline metrics identically across heads so
the comparison is apples-to-apples, and writes data/head_comparison.tsv.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from tisiago.caller import recall_at_fp_budget, reliability
from tisiago.scan_eval import grounding_stats

BUDGETS = (1.0, 5.0, 20.0)


def metrics_for(npz_path) -> list[dict]:
    """One metrics dict per ``p::`` head in a saved prediction file."""
    from sklearn.metrics import average_precision_score

    z = np.load(npz_path, allow_pickle=True)
    y = z["y"]
    cog = z["cognate"]
    tx = z["tx"]
    noncog = z["noncog"] if "noncog" in z.files else None
    yc, txc = y[cog], tx[cog]
    rows = []
    for key in [k for k in z.files if k.startswith("p::")]:
        p = z[key]
        pc = p[cog]
        row = {
            "head": key[3:],
            "source": Path(npz_path).name,
            "AUPRC": float(average_precision_score(yc, pc)),
        }
        thr = recall_at_fp_budget(pc, yc, txc, budget=1.0)["threshold"]
        for b in BUDGETS:
            row[f"recall@{int(b)}FP"] = float(
                recall_at_fp_budget(pc, yc, txc, budget=b)["recall"]
            )
        row["brier"] = float(reliability(pc, yc)["brier"])
        if noncog is not None:
            row["noncog_mean_p"] = float(grounding_stats(p[noncog], thr)["mean_p"])
        else:
            row["noncog_mean_p"] = float("nan")
        rows.append(row)
    return rows


def main() -> None:
    """Glob preds .npz, print the comparison table, write data/head_comparison.tsv."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="data/preds_*.npz")
    ap.add_argument("--out", default="data/head_comparison.tsv")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(args.glob)):
        rows.extend(metrics_for(f))
    if not rows:
        print(f"no prediction files matched {args.glob}")
        return

    cols = ["head", "AUPRC", "recall@1FP", "recall@5FP", "recall@20FP", "noncog_mean_p",
            "brier", "source"]
    hdr = "".join(f"{c:>16}" if c != "head" else f"{c:<20}" for c in cols)
    print(hdr)
    lines = ["\t".join(cols)]
    for r in sorted(rows, key=lambda d: -d["AUPRC"]):
        cells = [r["head"]] + [
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols[1:]
        ]
        print("".join(f"{c:>16}" if i else f"{c:<20}" for i, c in enumerate(cells)))
        lines.append("\t".join(cells))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
