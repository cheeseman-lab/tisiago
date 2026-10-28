"""Ribo-TISH start-site agreement at a matched number of predicted calls (K562 val, step 2).

Re-reads the existing ``predicted`` / ``real`` ``K562_TIS_predict_all.txt`` tables of each run's
``predicted_calls`` output (no Ribo-TISH rerun). Reference = the real-track calls after
:func:`ribotish.filter_predictions` (as in ``report.json``). Predicted candidates are ranked by
Ribo-TISH's combined ``FisherPvalue`` (TIS test x frame test; ties by ``TISPvalue``) or by the
TIS test alone (``TISPvalue``; ties by Fisher), one row per (Tid, Start), and the top N are
compared with :func:`predicted_calls.overlap` (the report's matching), overall, ATG /
near-cognate and by TisType, for two candidate pools:

- ``filtered``: only calls passing the report's filter (top N caps the call count);
- ``all``: every ``predict_all`` candidate (the rank replaces the filter's thresholds).

N = the real call count (``N_real``), and the largest N whose precision is at least the target
(``N_at_P``; default target: sb's 3-seed mean report precision). Writes the full curves to
``data/p6flux/matched_calls/``.

Usage:
    python autoresearch/p6flux/matched_calls.py [--models sb sbctx sbflux] [--target-precision P]
        [--condition K562] [--out data/p6flux/matched_calls]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func.predicted_calls import ATG, call_report, overlap
from tisiago.seq2func.ribotish import filter_predictions

ROOT = Path("data/p6/txtrain/export")
CALLS = "predicted_val_floatresample_s0/calls/K562"
TYPES = ("Annotated", "5'UTR")
OUT = Path("data/p6flux/matched_calls")


RANKS = {"fisher": ["FisherPvalue", "TISPvalue"], "tis": ["TISPvalue", "FisherPvalue"]}


def ranked(table: pd.DataFrame, rank: str = "fisher") -> pd.DataFrame:
    """Best row per (Tid, Start), sorted by :data:`RANKS` ``rank`` (ascending p values)."""
    t = table.sort_values(RANKS[rank], kind="mergesort")
    return t.drop_duplicates(["Tid", "Start"], keep="first").reset_index(drop=True)


def breakdown(predicted: pd.DataFrame, real: pd.DataFrame) -> dict:
    """:func:`overlap` overall, by start codon and for :data:`TYPES` (the report's grouping)."""

    def is_atg(df):
        return df.StartCodon.astype(str).str.upper() == ATG

    out = {"overall": overlap(predicted, real)}
    out["ATG"] = overlap(predicted[is_atg(predicted)], real[is_atg(real)])
    out["near_cognate"] = overlap(predicted[~is_atg(predicted)], real[~is_atg(real)])
    for t in TYPES:
        p, r = predicted.TisType.astype(str) == t, real.TisType.astype(str) == t
        out[t] = overlap(predicted[p], real[r])
    return out


def flat(stats: dict) -> dict:
    row = {}
    for group, v in stats.items():
        for k in ("jaccard", "recall", "precision"):
            row[f"{group}_{k[0].upper()}"] = v[k]
        row[f"{group}_n"] = v["n_observed"]
    return row


def curve(cands: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Overall precision / recall of the top N candidates for every N (set matching as overlap)."""
    keys = set(zip(real.Tid.astype(str), real.Start.astype(int), strict=True))
    hit = np.array(
        [
            (t, s) in keys
            for t, s in zip(cands.Tid.astype(str), cands.Start.astype(int), strict=True)
        ]
    )
    n = np.arange(1, len(cands) + 1)
    shared = np.cumsum(hit)
    return pd.DataFrame(
        {
            "n": n,
            "precision": shared / n,
            "recall": shared / len(keys),
            "jaccard": shared / (n + len(keys) - shared),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["sb", "sbctx", "sbflux"])
    ap.add_argument("--target-precision", type=float, default=None)
    ap.add_argument("--condition", default="K562")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    cond, out = args.condition, args.out
    table = f"{cond}_TIS_predict_all.txt"
    calls = CALLS.replace("K562", cond)
    runs = {}
    for model in args.models:
        for s in range(3):
            d = ROOT / f"{model}_{cond.lower()}_s{s}" / calls
            if (d / "predicted" / table).exists():
                runs[(model, s)] = d
    target = args.target_precision
    if target is None:
        target = float(
            np.mean(
                [
                    json.loads((d / "report.json").read_text())["overall"]["precision"]
                    for (m, _), d in runs.items()
                    if m == "sb"
                ]
            )
        )
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (model, seed), d in runs.items():
        pred_all = pd.read_csv(d / "predicted" / table, sep="\t")
        real_all = pd.read_csv(d / "real" / table, sep="\t")
        report = json.loads((d / "report.json").read_text())
        check = call_report(pred_all, real_all)["overall"]
        assert check["n_shared"] == report["overall"]["n_shared"], (model, seed)
        real = filter_predictions(real_all)
        n_real = len(set(zip(real.Tid, real.Start, strict=True)))
        base = {"model": model, "seed": seed, "n_real": n_real}
        rows.append(
            {**base, "setting": "report", **flat(breakdown(filter_predictions(pred_all), real))}
        )
        for rank in RANKS:
            pools = {
                "filtered": ranked(filter_predictions(pred_all), rank),
                "all": ranked(pred_all, rank),
            }
            for pool, cands in pools.items():
                cur = curve(cands, real)
                cur.to_csv(out / f"{model}_s{seed}_{rank}_{pool}_curve.tsv", sep="\t", index=False)
                ok = cur[cur.precision >= target]
                n_star = int(ok.n.max()) if len(ok) else 0
                for label, n in (("N_real", n_real), ("N_at_P", n_star)):
                    top = cands.head(n)
                    rows.append(
                        {
                            **base,
                            "setting": f"{rank}/{pool}/{label}",
                            "N": len(top),
                            **flat(breakdown(top, real)),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(out / "matched_calls.tsv", sep="\t", index=False)
    cols = [
        "overall_J",
        "overall_R",
        "overall_P",
        "overall_n",
        "ATG_J",
        "near_cognate_J",
        "Annotated_R",
        "Annotated_P",
        "5'UTR_R",
        "5'UTR_P",
        "5'UTR_n",
    ]
    pd.set_option("display.width", 250)
    print(f"target precision = {target:.3f} (sb 3-seed mean report precision)")
    for setting, g in df.groupby("setting", sort=False):
        mean = g.drop(columns=["seed", "setting"]).groupby("model", sort=False)
        m = mean.mean(numeric_only=True)
        m["J_sd"] = mean.overall_J.std()
        print(f"\n== {setting}")
        print(m[[c for c in ["N", *cols, "J_sd"] if c in m]].round(3).to_string())


if __name__ == "__main__":
    main()
