"""Deployed Ribo-TISH call cutoff for step-2 models: chosen on validation, applied once on test.

Deployed rule: a predicted call = passes :func:`ribotish.filter_predictions` (the report filter)
AND ``TISPvalue <= c``, with ``c <= TIS_P_MAX`` (``c = TIS_P_MAX`` is the report). Per model and
fold, ``c`` is chosen on each validation seed by

- ``count``: predicted call count = real call count (the TIS p of the N_real-th call ranked by
  TISPvalue; ``TIS_P_MAX`` if fewer calls pass the filter);
- ``precision``: the loosest c whose precision >= ``--target-precision`` (default 0.586, sb K562
  report precision; ``TIS_P_MAX`` if the filtered set already reaches it),

and the deployed cutoff is the geometric mean of the per-seed cutoffs (p values span orders of
magnitude). Test predictions (``export_test``, ``--split test`` Ribo-TISH outputs) are scored once
with that fixed cutoff, by :func:`predicted_calls.overlap` against the filtered real calls,
overall, ATG / near-cognate (:data:`predicted_calls.ATG`) and Annotated / 5'UTR.

Usage:
    python autoresearch/p6flux/deployed_cutoff.py [--models sb sbflux] [--folds k562 hela]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matched_calls import breakdown, flat, ranked  # sibling script

from tisiago.seq2func.predicted_calls import call_report
from tisiago.seq2func.ribotish import TIS_P_MAX, filter_predictions

CONDITION = {"k562": "K562", "hela": "HeLa", "u2os": "U2OS", "rpe1": "RPE1_Async"}
SPLITS = {
    "val": ("data/p6/txtrain/export", "predicted_val_floatresample_s0"),
    "test": ("data/p6/txtrain/export_test", "predicted_test_floatresample_s0"),
}
OUT = Path("data/p6flux/deployed_cutoff")


def tables(model: str, fold: str, seed: int, split: str):
    root, sub = SPLITS[split]
    d = Path(root) / f"{model}_{fold}_s{seed}" / sub / "calls" / CONDITION[fold]
    files = [d / k / f"{CONDITION[fold]}_TIS_predict_all.txt" for k in ("predicted", "real")]
    if not all(f.exists() for f in files):
        return None
    pred_all, real_all = (pd.read_csv(f, sep="\t") for f in files)
    report = d / "report.json"
    if report.exists():
        shared = json.loads(report.read_text())["overall"]["n_shared"]
        assert call_report(pred_all, real_all)["overall"]["n_shared"] == shared, d
    return pred_all, real_all


def deployed(pred_all: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    calls = filter_predictions(pred_all)
    return calls[calls.TISPvalue <= cutoff]


def choose(pred_all: pd.DataFrame, real: pd.DataFrame, target: float) -> dict[str, float]:
    cands = ranked(filter_predictions(pred_all), "tis")
    keys = set(zip(real.Tid.astype(str), real.Start.astype(int), strict=True))
    n_real = len(keys)
    p = cands.TISPvalue.to_numpy(float)
    hit = np.array(
        [
            (t, s) in keys
            for t, s in zip(cands.Tid.astype(str), cands.Start.astype(int), strict=True)
        ]
    )
    out = {"n_real": n_real, "n_filtered": len(cands)}
    out["count"] = float(p[n_real - 1]) if len(cands) > n_real else TIS_P_MAX
    prec = np.cumsum(hit) / np.arange(1, len(cands) + 1)
    # a cutoff keeps all ties at its p value: evaluate precision at the last row of each p value
    last = np.r_[p[1:] != p[:-1], True]
    ok = np.flatnonzero(last & (prec >= target))
    if len(ok) and ok[-1] == len(cands) - 1:
        out["precision"] = TIS_P_MAX
    else:
        out["precision"] = float(p[ok[-1]]) if len(ok) else float(p[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["sb", "sbflux"])
    ap.add_argument("--folds", nargs="+", default=["k562", "hela"])
    ap.add_argument("--target-precision", type=float, default=0.586)
    ap.add_argument("--val-only", action="store_true", help="choose cutoffs, do not read test")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cut_rows, test_rows = [], []
    for model in args.models:
        for fold in args.folds:
            chosen = []
            for seed in range(3):
                t = tables(model, fold, seed, "val")
                if t is None:
                    continue
                pred_all, real_all = t
                c = choose(pred_all, filter_predictions(real_all), args.target_precision)
                chosen.append(c)
                cut_rows.append({"model": model, "fold": fold, "seed": seed, **c})
            if not chosen:
                continue
            cutoffs = {
                rule: math.exp(np.mean([math.log(c[rule]) for c in chosen]))
                for rule in ("count", "precision")
            }
            cutoffs["report"] = TIS_P_MAX
            cut_rows.append({"model": model, "fold": fold, "seed": "geomean", **cutoffs})
            for seed in range(0 if args.val_only else 3):
                t = tables(model, fold, seed, "test")
                if t is None:
                    continue
                pred_all, real_all = t
                real = filter_predictions(real_all)
                for rule, c in cutoffs.items():
                    stats = breakdown(deployed(pred_all, c), real)
                    test_rows.append(
                        {
                            "model": model,
                            "fold": fold,
                            "seed": seed,
                            "rule": rule,
                            "cutoff": c,
                            **flat(stats),
                        }
                    )
    cuts = pd.DataFrame(cut_rows)
    cuts.to_csv(OUT / "cutoffs_val.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print("validation cutoffs (TISPvalue; geomean = deployed)")
    print(cuts.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    if args.val_only:
        return
    test = pd.DataFrame(test_rows)
    test.to_csv(OUT / "test_scores.tsv", sep="\t", index=False)
    cols = [
        "overall_J",
        "overall_R",
        "overall_P",
        "overall_n",
        "ATG_J",
        "ATG_R",
        "ATG_P",
        "near_cognate_J",
        "near_cognate_R",
        "near_cognate_P",
        "Annotated_R",
        "Annotated_P",
        "5'UTR_J",
        "5'UTR_R",
        "5'UTR_P",
        "5'UTR_n",
    ]
    print("\nTEST (per seed)")
    print(test[["model", "fold", "seed", "rule", *cols]].round(3).to_string(index=False))
    g = test.groupby(["fold", "rule", "model"], sort=False)
    mean = g[cols].mean()
    mean.insert(0, "seeds", g.size())
    mean["J_sd"] = g.overall_J.std()
    print("\nTEST (seed mean)")
    print(mean.round(3).to_string())


if __name__ == "__main__":
    main()
