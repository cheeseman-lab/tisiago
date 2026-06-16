"""Global all-codon caller: apply the calibrated head to the dense scan store.

Trains + isotonic-calibrates the head on the curated store (Phase 1's
``caller.fit_calibrated_head``), then scores every enumerated codon in the dense
scan store. Reports recall at a false-positives-per-transcript budget over
AUG+near-cognate codons *at true genome-wide imbalance* (all enumerated negatives,
not the curated 3:1), and the non-cognate grounding (mean P -> expect ~0).

Run after the scan store is assembled (GPU extraction + store.py). CPU only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.caller import (
    DEFAULT_KEYS,
    TRAIN_SUBSAMPLE,
    fit_calibrated_head,
    recall_at_fp_budget,
)


def grounding_stats(p, threshold: float) -> dict:
    """Summarize non-cognate scores: they should be ~0 for a grounded predictor.

    Args:
        p: probabilities for held-out non-cognate codons.
        threshold: the caller's operating threshold (for FPR).

    Returns:
        dict with ``mean_p``, ``p95``, ``fpr_at_threshold``, ``n``.
    """
    p = np.asarray(p, dtype=np.float64)
    return {
        "mean_p": float(p.mean()),
        "p95": float(np.percentile(p, 95)),
        "fpr_at_threshold": float((p >= threshold).mean()),
        "n": int(p.size),
    }


def _load(keys, emb):
    return np.concatenate([np.load(emb / k).astype(np.float32) for k in keys], axis=1)


def main() -> None:
    """Apply the curated-trained calibrated head to the dense scan store."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--curated-store", default="data/store")
    ap.add_argument("--scan-store", default="data/scan_store")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    ap.add_argument("--budget", type=float, default=1.0)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    cur = Path(args.curated_store)
    cm = pd.read_parquet(cur / "manifest.parquet")
    cy = cm.label_tis.values
    tr_all = np.where(cm.split.values == "train")[0]
    cal = np.where(cm.split.values == "val")[0]
    tr = rng.choice(tr_all, min(TRAIN_SUBSAMPLE, len(tr_all)), replace=False)
    Xc = _load(args.keys, cur / "embeddings")
    head = fit_calibrated_head(Xc[tr], cy[tr], Xc[cal], cy[cal])

    scan = Path(args.scan_store)
    sm = pd.read_parquet(scan / "manifest.parquet")
    Xs = _load(args.keys, scan / "embeddings")
    p = head["predict"](Xs)
    te = sm.split.values == "test"

    cognate = te & np.isin(sm.codon_class.values, ["AUG", "near_cognate"])
    noncog = te & (sm.codon_class.values == "non_cognate")
    ys = sm.label_tis.values
    res = recall_at_fp_budget(
        p[cognate], ys[cognate], sm.transcript_id.values[cognate], budget=args.budget
    )
    npos = int((ys[cognate] == 1).sum())
    ratio = (ys[cognate] == 0).sum() / max(1, npos)

    print(
        f"scan store: {len(sm)} positions  test cognate codons={int(cognate.sum())}  "
        f"non-cognate={int(noncog.sum())}"
    )
    print(f"true imbalance (neg:pos over cognate test codons) = {ratio:.1f}:1  (curated was 3:1)")
    print(f"\nCALLER @ <={args.budget} FP/transcript (true imbalance):")
    print(
        f"  recall={res['recall']:.3f}  threshold p>={res['threshold']:.3f}  "
        f"({res['fp_per_transcript']:.3f} FP/transcript)"
    )
    g = grounding_stats(p[noncog], threshold=res["threshold"])
    print("\nNON-COGNATE GROUNDING (held-out, never trained):")
    print(
        f"  mean p={g['mean_p']:.4f}  p95={g['p95']:.4f}  "
        f"FPR@threshold={g['fpr_at_threshold']:.4f}  (expect ~0)"
    )


if __name__ == "__main__":
    main()
