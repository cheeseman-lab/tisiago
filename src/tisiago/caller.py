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

from tisiago.manifest import split_grouped_indices, unique_site_indices


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
    if p.ndim != 1 or y.ndim != 1 or tx.ndim != 1:
        raise ValueError("p, y, and transcript_id must be one-dimensional")
    if not (len(p) == len(y) == len(tx)):
        raise ValueError("p, y, and transcript_id must have equal lengths")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not np.isfinite(p).all():
        raise ValueError("p must contain only finite scores")
    n_tx = len(np.unique(tx))
    n_pos = int(y.sum())

    best = {
        "recall": 0.0,
        "threshold": float("inf"),
        "fp_per_transcript": 0.0,
        "budget": float(budget),
    }
    if len(p) == 0 or n_pos == 0:
        return best

    # Sort once and evaluate only the ends of tied-score groups. The former
    # implementation allocated an N-element boolean mask at every distinct score,
    # making a dense scan O(N^2). This is O(N log N) and has identical >= semantics.
    order = np.argsort(-p, kind="stable")
    scores = p[order]
    labels = y[order]
    group_end = np.r_[scores[1:] != scores[:-1], True]
    end_idx = np.flatnonzero(group_end)
    tp = np.cumsum(labels, dtype=np.int64)[end_idx]
    fp = np.cumsum(~labels, dtype=np.int64)[end_idx]
    feasible = fp / max(1, n_tx) <= budget
    if feasible.any():
        max_tp = int(tp[feasible].max())
        # First match is the highest threshold for this recall (the documented tie-break).
        chosen = int(np.flatnonzero(feasible & (tp == max_tp))[0])
        best = {
            "recall": float(tp[chosen] / n_pos),
            "threshold": float(scores[end_idx[chosen]]),
            "fp_per_transcript": float(fp[chosen] / max(1, n_tx)),
            "budget": float(budget),
        }
    return best


def evaluate_at_threshold(p, y, transcript_id, threshold: float) -> dict:
    """Evaluate a fixed, previously selected operating threshold.

    Unlike :func:`recall_at_fp_budget`, this function does not optimize anything on
    the supplied data. Use it on test data with a threshold selected on validation
    data to estimate deployable performance without test-set threshold leakage.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    tx = np.asarray(transcript_id)
    if p.ndim != 1 or y.ndim != 1 or tx.ndim != 1:
        raise ValueError("p, y, and transcript_id must be one-dimensional")
    if not (len(p) == len(y) == len(tx)):
        raise ValueError("p, y, and transcript_id must have equal lengths")
    if not np.isfinite(p).all():
        raise ValueError("p must contain only finite scores")

    admitted = p >= threshold
    n_pos = int(y.sum())
    n_tx = len(np.unique(tx))
    tp = int((admitted & y).sum())
    fp = int((admitted & ~y).sum())
    return {
        "recall": float(tp / max(1, n_pos)),
        "threshold": float(threshold),
        "fp_per_transcript": float(fp / max(1, n_tx)),
        "true_positives": tp,
        "false_positives": fp,
    }


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


def fit_calibrated_head(
    X_train, y_train, X_cal, y_cal, *, C: float = 1.0, max_iter: int = 300, class_weight=None
) -> dict:
    """Train a standardized logistic head, then isotonic-calibrate on a held-out set.

    Args:
        X_train: training features, shape [N_train, D].
        y_train: training labels, shape [N_train].
        X_cal: held-out calibration features (a different chromosome split), shape [N_cal, D].
        y_cal: held-out calibration labels, shape [N_cal].
        C: inverse L2 strength for the logistic head.
        max_iter: lbfgs iteration cap.
        class_weight: ``None`` | ``"balanced"`` | dict — passed to ``LogisticRegression``.

    Returns:
        dict with ``predict`` (X -> calibrated p) and ``predict_raw`` (X -> uncalibrated p).
    """
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight=class_weight).fit(
        scaler.transform(X_train), y_train
    )

    # Fold StandardScaler into the linear layer once. At inference this replaces
    # scaler.transform(X) + sklearn validation + predict_proba with one matrix-vector
    # product, without changing the fitted decision function:
    #   ((X - mean) / scale) @ beta + b == X @ (beta / scale) + adjusted_b
    linear_coef = (clf.coef_[0] / scaler.scale_).astype(np.float32)
    linear_intercept = float(clf.intercept_[0] - scaler.mean_ @ linear_coef)

    def predict_logit(X):
        return np.asarray(X) @ linear_coef + linear_intercept

    def predict_from_logit(logit):
        logit = np.asarray(logit)
        # Stable sigmoid without adding scipy as a direct runtime dependency.
        out = np.empty_like(logit, dtype=np.float64)
        positive = logit >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
        exp_x = np.exp(logit[~positive])
        out[~positive] = exp_x / (1.0 + exp_x)
        return out

    def predict_raw(X):
        return predict_from_logit(predict_logit(X))

    iso = IsotonicRegression(out_of_bounds="clip").fit(predict_raw(X_cal), y_cal)

    def calibrate_raw(p):
        return iso.predict(np.asarray(p))

    def predict(X):
        return calibrate_raw(predict_raw(X))

    return {
        "predict": predict,
        "predict_raw": predict_raw,
        "predict_logit": predict_logit,
        "predict_from_logit": predict_from_logit,
        "calibrate_raw": calibrate_raw,
        "linear_coef": linear_coef,
        "linear_intercept": linear_intercept,
        "model": clf,
        "scaler": scaler,
        "calibrator": iso,
    }


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

    unique = unique_site_indices(m)
    tr_all = unique[m.split.values[unique] == "train"]
    val = unique[m.split.values[unique] == "val"]
    te = unique[m.split.values[unique] == "test"]
    cal, operating = split_grouped_indices(
        val, m.transcript_id.values[val], fraction=0.5, seed=0
    )
    tr = rng.choice(tr_all, min(TRAIN_SUBSAMPLE, len(tr_all)), replace=False)

    X = load(args.keys, emb)
    head = fit_calibrated_head(X[tr], y[tr], X[cal], y[cal])
    p_val = head["predict"](X[operating])
    p_raw = head["predict_raw"](X[te])
    p_cal = head["predict"](X[te])
    yte = y[te]
    tx_te = m.transcript_id.values[te]

    print(f"features: {'+'.join(args.keys)}  dim={X.shape[1]}")
    print(
        f"train={len(tr)} (of {len(tr_all)})  calibration={len(cal)}  "
        f"operating-val={len(operating)}  test={len(te)}  "
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
    selected = recall_at_fp_budget(
        p_val,
        y[operating],
        m.transcript_id.values[operating],
        budget=args.budget,
    )
    res = evaluate_at_threshold(p_cal, yte, tx_te, selected["threshold"])
    print(
        f"  TEST recall={res['recall']:.3f} at VAL-selected threshold "
        f"p≥{res['threshold']:.3f} ({res['fp_per_transcript']:.3f} TEST FP/transcript)"
    )
    oracle = recall_at_fp_budget(p_cal, yte, tx_te, budget=args.budget)
    print(f"  diagnostic oracle TEST recall={oracle['recall']:.3f} (threshold fit on TEST)")
    print("\n  NOTE: on the curated 3:1 decoy set, NOT true genome-wide imbalance.")
    print("  The true-imbalance number requires the Phase 2 dense codon scan.")


if __name__ == "__main__":
    main()
