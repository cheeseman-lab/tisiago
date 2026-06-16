"""Autoresearch experiment: train a head on the 1:3 balanced set, write predictions.

The autoresearch agent edits ONLY the CONFIG block below — feature subset, head type,
hyperparameters, class weighting, and the train-negative subsample ratio (within the 1:3
pool). It then runs ``evaluate.py`` (fixed) which scores ``preds.npz`` on all four metrics.

Reads the embedding store via an absolute ``--store`` path so the four parallel worktrees
share one read-only copy. Trains on ``train``, predicts on ``val`` + ``test`` (the loop
climbs a val metric and reports test — never trains/selects on test).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ===================== CONFIG — autoresearch edits this block =====================
# Feature .npy keys to concatenate (subset of the 14 in data/store/config.yaml).
FEATURE_KEYS = [
    "alphagenome_jax/L16k/decoder_1bp/off0.npy",
    "evo2/W8k/blocks.28.mlp.l3/off0.npy",
]
HEAD = "logistic"  # "logistic" | "mlp"
HEAD_PARAMS = {"C": 1.0}  # logistic: C ; mlp: hidden_layer_sizes, alpha, ...
CLASS_WEIGHT = None  # None | "balanced" | {0: w0, 1: w1}  (logistic only)
NEG_SUBSAMPLE = None  # None = all train negatives ; else float neg:pos ratio
SEED = 0
TRAIN_CAP = 60000  # cap train rows for speed (matches eval.py/caller.py)
# =================================================================================


def load(keys, emb):
    """Concatenate the given .npy feature arrays along the feature axis."""
    return np.concatenate([np.load(emb / k).astype(np.float32) for k in keys], axis=1)


def _subsample_train(tr_idx, y, neg_ratio, rng):
    """Optionally thin train negatives to a target neg:pos ratio (keep all positives)."""
    if neg_ratio is None:
        return tr_idx
    pos = tr_idx[y[tr_idx] == 1]
    neg = tr_idx[y[tr_idx] == 0]
    n_neg = min(len(neg), int(round(neg_ratio * len(pos))))
    neg = rng.choice(neg, n_neg, replace=False)
    return np.concatenate([pos, neg])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, help="Absolute path to the assembled store.")
    ap.add_argument("--out", default="preds.npz")
    # Optional overrides for baseline sweeps. The autoresearch agent edits the CONFIG block
    # and passes neither; these just let us run reference floors without touching the file.
    ap.add_argument("--keys", nargs="+", default=None, help="Override FEATURE_KEYS.")
    ap.add_argument("--head", default=None, choices=["logistic", "mlp"], help="Override HEAD.")
    args = ap.parse_args()

    feature_keys = args.keys if args.keys is not None else FEATURE_KEYS
    head = args.head if args.head is not None else HEAD

    store = Path(args.store)
    emb = store / "embeddings"
    m = pd.read_parquet(store / "manifest.parquet")
    y = m.label_tis.values
    rng = np.random.default_rng(SEED)

    tr_all = np.where(m.split.values == "train")[0]
    va = np.where(m.split.values == "val")[0]
    te = np.where(m.split.values == "test")[0]
    if len(tr_all) > TRAIN_CAP:
        tr_all = rng.choice(tr_all, TRAIN_CAP, replace=False)
    tr = _subsample_train(tr_all, y, NEG_SUBSAMPLE, rng)

    X = load(feature_keys, emb)
    sc = StandardScaler().fit(X[tr])
    Xtr = sc.transform(X[tr])

    if head == "logistic":
        clf = LogisticRegression(max_iter=1000, class_weight=CLASS_WEIGHT, **HEAD_PARAMS)
    elif head == "mlp":
        clf = MLPClassifier(random_state=SEED, early_stopping=True, **HEAD_PARAMS)
    else:
        raise SystemExit(f"unknown HEAD {head!r}")
    clf.fit(Xtr, y[tr])

    def predict(idx):
        return clf.predict_proba(sc.transform(X[idx]))[:, 1]

    # strand-aware plus-strand codon position, for the near-neighbour win-rate metric.
    a_pos = np.where(m.strand.values == "+", m.gstart.values, m.gstart.values - 1)
    np.savez(
        args.out,
        p_val=predict(va),
        y_val=y[va],
        tx_val=m.transcript_id.values[va].astype(str),
        apos_val=a_pos[va],
        p_test=predict(te),
        y_test=y[te],
        tx_test=m.transcript_id.values[te].astype(str),
        apos_test=a_pos[te],
    )
    print(
        f"config: dim={X.shape[1]} keys={len(feature_keys)} head={head} params={HEAD_PARAMS} "
        f"class_weight={CLASS_WEIGHT} neg_subsample={NEG_SUBSAMPLE} n_train={len(tr)}"
    )


if __name__ == "__main__":
    main()
