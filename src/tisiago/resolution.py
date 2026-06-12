"""Is the TIS signal at true single-nucleotide resolution, or coarse context?

Trains a logistic head on the train chromosomes, then on the held-out test set
asks: for each real TIS, can the head rank it ABOVE a decoy candidate codon in
the SAME transcript within D bp? Long-range/regional context is useless here, so
a high pairwise win-rate means genuine nucleotide-level discrimination.

AlphaGenome's 1bp embedding is upsampled from a 128bp trunk, so we expect it to
struggle more than Evo2 (true 1-token-per-bp) at small D — that contrast is the
point.
"""

from __future__ import annotations

from pathlib import Path

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)

MODELS = {
    "AlphaGenome 16k": ["alphagenome_jax/L16k/decoder_1bp/off0.npy"],
    "Evo2 blk28 off0": ["evo2/W8k/blocks.28.mlp.l3/off0.npy"],
    "AG16k + Evo2 blk28": ["alphagenome_jax/L16k/decoder_1bp/off0.npy", "evo2/W8k/blocks.28.mlp.l3/off0.npy"],
}
DISTANCES = [64, 128, 512, 2000]


def load(keys, emb):
    return np.concatenate([np.load(emb / k).astype(np.float32) for k in keys], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="data/store", help="Path to the assembled vector store.")
    args = ap.parse_args()
    store = Path(args.store)
    emb = store / "embeddings"
    m = pd.read_parquet(store / "manifest.parquet")
    y = m.label_tis.values
    tr_all = np.where(m.split.values == "train")[0]
    te = np.where(m.split.values == "test")[0]
    tr = RNG.choice(tr_all, min(60000, len(tr_all)), replace=False)

    mte = m.iloc[te].reset_index(drop=True)
    # candidate genomic A position (strand-aware), and group by transcript
    a_pos = np.where(mte.strand.values == "+", mte.gstart.values, mte.gstart.values - 1)
    by_tx: dict[str, list[int]] = {}
    for i, t in enumerate(mte.transcript_id.values):
        by_tx.setdefault(t, []).append(i)

    # Precompute, per distance, the list of (positive_idx, decoy_idx) near pairs.
    pairs_by_d = {d: [] for d in DISTANCES}
    yte = y[te]
    for idxs in by_tx.values():
        idxs = np.array(idxs)
        pos = idxs[yte[idxs] == 1]
        neg = idxs[yte[idxs] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        for p_i in pos:
            dist = np.abs(a_pos[neg] - a_pos[p_i])
            for d in DISTANCES:
                near = neg[dist <= d]
                for n_i in near:
                    pairs_by_d[d].append((p_i, n_i))

    print(f"test candidates={len(te)}  transcripts={len(by_tx)}")
    print("near-neighbour decoy pairs per distance:",
          {d: len(pairs_by_d[d]) for d in DISTANCES}, "\n")
    print(f"{'model':22s} " + " ".join(f"win@{d}bp" for d in DISTANCES) + "   all-AUROC")
    print("-" * 70)

    from sklearn.metrics import roc_auc_score
    for name, keys in MODELS.items():
        X = load(keys, emb)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=300).fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        auc = roc_auc_score(yte, p)
        wins = []
        for d in DISTANCES:
            pr = pairs_by_d[d]
            if not pr:
                wins.append("   -  ")
                continue
            pi = np.fromiter((a for a, _ in pr), int)
            ni = np.fromiter((b for _, b in pr), int)
            wr = np.mean(p[pi] > p[ni])  # fraction real-TIS ranked above near decoy
            wins.append(f"{wr:.3f} ")
        print(f"{name:22s} " + "  ".join(wins) + f"     {auc:.3f}")
        del X
    print("\nwin@Dbp = P(real TIS scored above a decoy codon within D bp, same transcript).")
    print("0.5 = no nucleotide-level discrimination; ->1.0 = sharp single-base resolution.")


if __name__ == "__main__":
    main()
