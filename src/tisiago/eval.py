"""Quick held-out comparison of feature sets for TIS yes/no prediction.

Simple heads (standardized logistic + one small MLP) trained on the `train`
chromosomes, evaluated on the held-out `test` chromosomes. Reports AUROC and
AUPRC (positive-rate baseline shown for AUPRC context). Not a real sweep — a
sanity check of which frozen embeddings carry TIS signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from tisiago.manifest import unique_site_indices

RNG = np.random.default_rng(0)
TRAIN_SUBSAMPLE = 60000  # keep logistic/MLP fast; test always uses the full split

FEATURE_SETS = {
    "AlphaGenome 16k": ["alphagenome_jax/L16k/decoder_1bp/off0.npy"],
    "AlphaGenome 131k": ["alphagenome_jax/L131k/decoder_1bp/off0.npy"],
    "Evo2 blk28 off0": ["evo2/W8k/blocks.28.mlp.l3/off0.npy"],
    "AG16k + Evo2 blk28 off0": [
        "alphagenome_jax/L16k/decoder_1bp/off0.npy",
        "evo2/W8k/blocks.28.mlp.l3/off0.npy",
    ],
}


def load(keys, emb):
    """Load and concatenate a feature-key set."""
    return np.concatenate([np.load(emb / k).astype(np.float32) for k in keys], axis=1)


def main():
    """Compare frozen feature sets on unique held-out candidate sites."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="data/store", help="Path to the assembled vector store.")
    ap.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Evaluate one concatenation of .npy keys instead of the default sweep.",
    )
    args = ap.parse_args()
    store = Path(args.store)
    emb = store / "embeddings"
    feature_sets = {"+".join(args.keys): args.keys} if args.keys else FEATURE_SETS

    m = pd.read_parquet(store / "manifest.parquet")
    y = m.label_tis.values
    unique = unique_site_indices(m)
    tr_all = unique[m.split.values[unique] == "train"]
    va = unique[m.split.values[unique] == "val"]
    te = unique[m.split.values[unique] == "test"]
    tr = RNG.choice(tr_all, min(TRAIN_SUBSAMPLE, len(tr_all)), replace=False)
    pos_rate = y[te].mean()
    print(
        f"train={len(tr)} (subsampled from {len(tr_all)})  test={len(te)}  "
        f"test positive-rate={pos_rate:.3f}\n"
    )
    print(
        f"{'feature set':26s} {'dim':>6} {'val AUROC':>10} "
        f"{'test AUROC':>11} {'test AUPRC':>11}"
    )
    print("-" * 76)

    best_key, best = None, None
    for name, keys in feature_sets.items():
        X = load(keys, emb)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(Xtr, y[tr])
        p_val = clf.predict_proba(Xva)[:, 1]
        p = clf.predict_proba(Xte)[:, 1]
        val_auc = roc_auc_score(y[va], p_val)
        auc, ap = roc_auc_score(y[te], p), average_precision_score(y[te], p)
        print(f"{name:26s} {X.shape[1]:>6} {val_auc:>10.4f} {auc:>11.4f} {ap:>11.3f}")
        if best is None or val_auc > best:
            best, best_key, best_sc, best_X = val_auc, name, sc, X
        del X

    # one small MLP on the best logistic feature set, to see if nonlinearity helps
    print("-" * 76)
    Xtr, Xte = best_sc.transform(best_X[tr]), best_sc.transform(best_X[te])
    mlp = MLPClassifier(
        hidden_layer_sizes=(256,), max_iter=60, early_stopping=True, random_state=0
    )
    mlp.fit(Xtr, y[tr])
    p_mlp = mlp.predict_proba(Xte)[:, 1]
    print(
        f"MLP(256) on '{best_key}':  test "
        f"AUROC={roc_auc_score(y[te], p_mlp):.4f}  "
        f"AUPRC={average_precision_score(y[te], p_mlp):.3f}"
    )
    print(f"(AUPRC baseline = test positive-rate = {pos_rate:.3f})")

    # ---- THE KEY CHECK: is the signal just recognizing canonical annotated starts? ----
    # Refit the best logistic, then score the held-out set within strata. A subset's
    # AUROC = "can the head rank THESE positives above the negatives". The honest,
    # novel question is whether ALTERNATIVE (non-canonical) TIS still separate.
    print("\n" + "=" * 60)
    print(f"STRATIFIED held-out AUROC  ('{best_key}', logistic)")
    print("=" * 60)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(best_sc.transform(best_X[tr]), y[tr])
    p = clf.predict_proba(best_sc.transform(best_X[te]))[:, 1]
    yte = y[te]
    m_te = m.iloc[te].reset_index(drop=True)
    neg = yte == 0

    def stratum_auc(pos_mask, label):
        keep = pos_mask | neg
        npos = int(pos_mask.sum())
        if npos < 20:
            print(f"  {label:32s} n_pos={npos:>5}  (too few)")
            return
        auc = roc_auc_score(yte[keep], p[keep])
        print(f"  {label:32s} n_pos={npos:>5}  AUROC={auc:.4f}")

    canon = (yte == 1) & m_te.is_canonical.values
    noncanon = (yte == 1) & ~m_te.is_canonical.values
    stratum_auc(yte == 1, "ALL positives (headline)")
    stratum_auc(canon, "canonical annotated starts")
    stratum_auc(noncanon, "NON-canonical (alt TIS) <-- key")
    for rc in ["uORF_5UTR", "dTIS_CDS", "extension_5UTR"]:
        stratum_auc((yte == 1) & (m_te.region_class.values == rc), f"  region={rc}")


if __name__ == "__main__":
    main()
