"""Calibrated, caller-shaped evaluation of the TIS head (deliverable track).

Where ``eval.py`` / ``resolution.py`` measure *ranking* (AUROC, win-rate), this
module measures whether the head emits a *calibrated probability* and *rejects*
non-starts: reliability + Brier, and recall at a false-positives-per-transcript
budget. Metric functions take plain arrays so the Phase-2 dense scan reuses them
unchanged.

Train on the ``train`` split, calibrate on the held-out ``val`` split (chr7),
evaluate on ``test`` (chr8/chr9). Pure CPU over an assembled store.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler


def recall_at_fp_budget(p, y, transcript_id, budget: float = 1.0) -> dict:
    """Recall at the threshold whose mean false-positives-per-transcript ≤ budget.

    A false positive is a negative codon scored ≥ threshold. The budget is the mean,
    over distinct transcripts, of such false positives. We pick the *tightest* (highest)
    threshold that reaches the maximum recall still satisfying the budget.

    Args:
        p: calibrated probabilities, shape [N].
        y: binary labels (1 = called TIS), shape [N].
        transcript_id: per-row transcript id, shape [N].
        budget: maximum allowed mean false positives per transcript.

    Returns:
        dict with ``recall``, ``threshold``, ``fp_per_transcript``, ``budget``.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    tx = np.asarray(transcript_id)
    n_tx = len(np.unique(tx))
    n_pos = int(y.sum())

    # Candidate thresholds: all unique scores plus an +inf sentinel (admit nothing).
    # Sweep from strict (high) to loose (low); each score used as a ">=" cutoff so
    # every distinct operating point is visited.
    candidates = np.concatenate([np.unique(p), [np.inf]])
    candidates.sort()  # ascending
    candidates = candidates[::-1]  # descending: strict -> loose

    best = {
        "recall": 0.0,
        "threshold": float("inf"),
        "fp_per_transcript": 0.0,
        "budget": float(budget),
    }
    for thr in candidates:
        admitted = p >= thr
        fp = admitted & ~y
        # mean FP per transcript across ALL transcripts present (not just those with FP)
        fp_per_tx = fp.sum() / max(1, n_tx)
        if fp_per_tx <= budget:
            recall = (admitted & y).sum() / max(1, n_pos)
            if recall > best["recall"]:  # strict: keep the tightest threshold at each recall level
                best = {
                    "recall": float(recall),
                    "threshold": float(thr),
                    "fp_per_transcript": float(fp_per_tx),
                    "budget": float(budget),
                }
        else:
            break  # thresholds only get looser -> FP only grows
    return best


def reliability(p, y, n_bins: int = 10) -> dict:
    """Binned calibration curve + Brier score.

    Args:
        p: probabilities, shape [N].
        y: binary labels, shape [N].
        n_bins: number of equal-width bins over [0, 1].

    Returns:
        dict with ``brier``, ``max_gap`` (max |mean_p - mean_y| over non-empty bins),
        ``bin_confidence`` (mean p per bin), ``bin_accuracy`` (mean y per bin); empty
        bins report NaN.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf[b] = p[m].mean()
            acc[b] = y[m].mean()
    gaps = np.abs(conf - acc)
    return {
        "brier": float(brier_score_loss(y, p)),
        "max_gap": float(np.nanmax(gaps)) if np.isfinite(gaps).any() else float("nan"),
        "bin_confidence": conf.tolist(),
        "bin_accuracy": acc.tolist(),
    }


def fit_calibrated_head(X_train, y_train, X_cal, y_cal) -> dict:
    """Train a standardized logistic head, then isotonic-calibrate on a held-out set.

    Args:
        X_train: training features, shape [N_train, D].
        y_train: training labels, shape [N_train].
        X_cal: held-out calibration features (a different chromosome split), shape [N_cal, D].
        y_cal: held-out calibration labels, shape [N_cal].

    Returns:
        dict with ``predict`` (X -> calibrated p) and ``predict_raw`` (X -> uncalibrated p).
    """
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(scaler.transform(X_train), y_train)

    def predict_raw(X):
        return clf.predict_proba(scaler.transform(X))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(predict_raw(X_cal), y_cal)

    def predict(X):
        return iso.predict(predict_raw(X))

    return {"predict": predict, "predict_raw": predict_raw}


# default feature set = the FINDINGS headline (AlphaGenome 16k + Evo2 blk28).
DEFAULT_KEYS = [
    "alphagenome_jax/L16k/decoder_1bp/off0.npy",
    "evo2/W8k/blocks.28.mlp.l3/off0.npy",
]
TRAIN_SUBSAMPLE = 60000


def load(keys, emb):
    """Concatenate the given .npy feature arrays along the feature axis."""
    return np.concatenate([np.load(emb / k).astype(np.float32) for k in keys], axis=1)


def main() -> None:
    """Run calibration + caller-metric report on the assembled store."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--store", default="data/store", help="Path to the assembled store.")
    ap.add_argument(
        "--keys",
        nargs="+",
        default=DEFAULT_KEYS,
        help="Feature .npy keys to concatenate (default: AG16k + Evo2 blk28).",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=1.0,
        help="Max mean false positives per transcript for the headline metric.",
    )
    args = ap.parse_args()

    store = Path(args.store)
    emb = store / "embeddings"
    m = pd.read_parquet(store / "manifest.parquet")
    y = m.label_tis.values
    rng = np.random.default_rng(0)

    tr_all = np.where(m.split.values == "train")[0]
    cal = np.where(m.split.values == "val")[0]
    te = np.where(m.split.values == "test")[0]
    tr = rng.choice(tr_all, min(TRAIN_SUBSAMPLE, len(tr_all)), replace=False)

    X = load(args.keys, emb)
    head = fit_calibrated_head(X[tr], y[tr], X[cal], y[cal])
    p_raw = head["predict_raw"](X[te])
    p_cal = head["predict"](X[te])
    yte = y[te]
    tx_te = m.transcript_id.values[te]

    print(f"features: {'+'.join(args.keys)}  dim={X.shape[1]}")
    print(
        f"train={len(tr)} (of {len(tr_all)})  cal/val={len(cal)}  test={len(te)}  "
        f"test transcripts={len(np.unique(tx_te))}  test pos-rate={yte.mean():.3f}"
    )
    print(
        f"\nAUPRC (ranking, calibration-invariant)={average_precision_score(yte, p_cal):.3f}  "
        f"(baseline = test pos-rate = {yte.mean():.3f})"
    )
    print("\n-- calibration (held-out test) --")
    for name, p in [("raw logistic", p_raw), ("isotonic-calibrated", p_cal)]:
        r = reliability(p, yte, n_bins=10)
        print(f"  {name:22s} Brier={r['brier']:.4f}  max_gap={r['max_gap']:.3f}")

    print(f"\n-- caller metric (calibrated, budget ≤ {args.budget} FP/transcript) --")
    res = recall_at_fp_budget(p_cal, yte, tx_te, budget=args.budget)
    print(
        f"  recall={res['recall']:.3f}  at threshold p≥{res['threshold']:.3f}  "
        f"(actual {res['fp_per_transcript']:.3f} FP/transcript)"
    )
    print("\n  NOTE: on the curated 3:1 decoy set, NOT true genome-wide imbalance.")
    print("  The true-imbalance number requires the Phase 2 dense codon scan.")


if __name__ == "__main__":
    main()
