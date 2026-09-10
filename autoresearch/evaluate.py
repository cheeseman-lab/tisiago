"""Fixed evaluation harness for the autoresearch fleet — NEVER edited by the agent.

Loads ``preds.npz`` (written by train_experiment.py) and computes all four objective
metrics on BOTH val and test:

* ``auprc``      — average precision (precision-aware ranking)
* ``auroc``      — ROC AUC (overall ranking)
* ``recall1fp``  — recall at <=1 false positive / transcript (caller operating point)
* ``winrate64``  — near-neighbour win-rate @64bp (hard base-resolution discrimination)

It prints grep-friendly ``key: value`` lines, then echoes the val value of the metric named
by the ``OBJECTIVE`` env var as the canonical ``objective:`` line each loop climbs. The four
parallel loops differ only in ``OBJECTIVE`` (auprc / auroc / recall1fp / winrate64).

The win-rate reuses the near-neighbour decoy-pairing logic from ``src/tisiago/resolution.py``
(real TIS vs a decoy codon within D bp in the same transcript), computed here on val too.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from tisiago.caller import evaluate_at_threshold, recall_at_fp_budget

OBJECTIVES = ("auprc", "auroc", "recall1fp", "winrate64")
WIN_DISTANCE = 64


def winrate(p, y, tx, a_pos, d=WIN_DISTANCE):
    """P(real TIS scored above a decoy codon within ``d`` bp in the same transcript)."""
    y = np.asarray(y).astype(bool)
    by_tx: dict[str, list[int]] = {}
    for i, t in enumerate(tx):
        by_tx.setdefault(t, []).append(i)
    wins = total = 0
    for idxs in by_tx.values():
        idxs = np.asarray(idxs)
        pos = idxs[y[idxs]]
        neg = idxs[~y[idxs]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        for pi in pos:
            near = neg[np.abs(a_pos[neg] - a_pos[pi]) <= d]
            wins += int(np.sum(p[near] < p[pi]))
            total += len(near)
    return wins / total if total else float("nan")


def metrics(p, y, tx, a_pos) -> dict:
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "recall1fp": float(recall_at_fp_budget(p, y, tx, budget=1.0)["recall"]),
        "winrate64": float(winrate(p, y, tx, a_pos)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="preds.npz")
    args = ap.parse_args()

    z = np.load(args.preds, allow_pickle=True)
    mv = metrics(z["p_val"], z["y_val"], z["tx_val"], z["apos_val"])
    mt = metrics(z["p_test"], z["y_test"], z["tx_test"], z["apos_test"])
    oracle_test_recall = mt["recall1fp"]
    selected = recall_at_fp_budget(
        z["p_val"], z["y_val"], z["tx_val"], budget=1.0
    )
    mt["recall1fp"] = evaluate_at_threshold(
        z["p_test"], z["y_test"], z["tx_test"], selected["threshold"]
    )["recall"]

    for k in OBJECTIVES:
        print(f"{k}_val: {mv[k]:.4f}")
    for k in OBJECTIVES:
        print(f"{k}_test: {mt[k]:.4f}")
    print(f"recall1fp_oracle_test: {oracle_test_recall:.4f}")

    objective = os.environ.get("OBJECTIVE", "auprc")
    if objective not in OBJECTIVES:
        raise SystemExit(f"OBJECTIVE={objective!r} not in {OBJECTIVES}")
    print(f"objective: {mv[objective]:.4f}  ({objective}_val)")


if __name__ == "__main__":
    main()
