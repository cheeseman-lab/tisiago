"""Option B — retrain the head to be imbalance-aware, evaluate at true genome-wide imbalance.

The curated head ranks 3:1 decoys at 0.90 AUROC but collapses to 0.094 recall at the real
~230:1 imbalance (FINDINGS §5). Option B keeps the autoresearch-found 7-key feature stack and
heavy L2 but adds ``class_weight="balanced"`` so the decision boundary is set for imbalance.

This module trains TWO calibrated heads on the **curated** store (192k) and scores them on the
**dense scan** store's TEST split at true imbalance:

* **Config C** — autoresearch best-all-rounder: ``C=0.00075, class_weight=None``.
* **Option B** — same stack/regularization, ``class_weight="balanced"``.

Reports, side by side: recall @ ≤budget FP/transcript over cognate (AUG+near_cognate) test
codons, the non-cognate grounding (mean p → expect ~0), and reliability/Brier. ``score_demo``
scores every cognate codon of an out-of-sample gene set (SCN1A/GRIN1/TSC1).

The 7-key store is ~196 GB fp16 over the 5M test rows, so predictions run **chunked over a
memmap'd store** — never materializing the full matrix. Pure CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from tisiago.caller import fit_calibrated_head, recall_at_fp_budget, reliability
from tisiago.head_xgb import fit_lgb_head, fit_rf_head, fit_xgb_head
from tisiago.scan_eval import grounding_stats

# The autoresearch winner stack (see autoresearch/winners.md).
KEYS_7 = [
    "alphagenome_jax/L16k/decoder_1bp/off0.npy",
    "alphagenome_jax/L131k/decoder_1bp/off0.npy",
    "evo2/W8k/blocks.28.mlp.l3/off0.npy",
    "evo2/W8k/blocks.28.mlp.l3/off3.npy",
    "evo2/W8k/blocks.28.mlp.l3/off6.npy",
    "evo2/W8k/blocks.28.mlp.l3/off9.npy",
    "onehot/kozakW20.npy",
]
# AG + one-hot subset — the substrate available before the genome-wide evo2 scan finishes.
KEYS_AG = [
    "alphagenome_jax/L16k/decoder_1bp/off0.npy",
    "alphagenome_jax/L131k/decoder_1bp/off0.npy",
    "onehot/kozakW20.npy",
]
FEATURE_SETS = {"ag7": KEYS_7, "ag": KEYS_AG}
TRAIN_SUBSAMPLE = 60000


def _load_full(emb: Path, keys, rows=None) -> np.ndarray:
    """Load feature keys into a single float32 [N, D]; optional row subset.

    Fills a preallocated output column-block by column-block rather than building a
    parts list and concatenating — the concatenate would hold both the parts and the
    result at once (~2x the array in RAM). Result is identical to the concatenate.
    """
    mmaps = [np.load(emb / k, mmap_mode="r") for k in keys]
    n = mmaps[0].shape[0] if rows is None else len(rows)
    d = sum(a.shape[1] for a in mmaps)
    out = np.empty((n, d), dtype=np.float32)
    c = 0
    for a in mmaps:
        w = a.shape[1]
        out[:, c : c + w] = a[:] if rows is None else a[rows]
        c += w
    return out


def _predict_chunked(emb: Path, keys, heads: dict, row_idx: np.ndarray, chunk: int = 100_000):
    """Predict each head over ``row_idx`` of a (large) memmap'd store, in chunks.

    Returns ``{head_name: p}`` with ``p`` aligned to the order of ``row_idx``.
    """
    mmaps = [np.load(emb / k, mmap_mode="r") for k in keys]
    out = {name: np.empty(len(row_idx), dtype=np.float64) for name in heads}
    for s in range(0, len(row_idx), chunk):
        sel = row_idx[s : s + chunk]
        X = np.concatenate([np.asarray(a[sel], dtype=np.float32) for a in mmaps], axis=1)
        for name, head in heads.items():
            out[name][s : s + chunk] = head["predict"](X)
        print(f"    scored {min(s + chunk, len(row_idx)):,}/{len(row_idx):,}", flush=True)
    return out


def train_heads(curated_store: Path, keys=KEYS_7, only=None) -> dict:
    """Train Config C and Option B calibrated heads on the curated store.

    ``only`` restricts to a subset of spec names (e.g. just the baseline used in dense
    mode) — avoids fitting the three unused heads on a 19.6k-dim stack.
    """
    emb = curated_store / "embeddings"
    m = pd.read_parquet(curated_store / "manifest.parquet")
    y = m.label_tis.values
    rng = np.random.default_rng(0)
    tr_all = np.where(m.split.values == "train")[0]
    cal = np.where(m.split.values == "val")[0]
    tr = rng.choice(tr_all, min(TRAIN_SUBSAMPLE, len(tr_all)), replace=False)

    X = _load_full(emb, keys)
    print(f"curated: train={len(tr)} cal/val={len(cal)} dim={X.shape[1]}", flush=True)
    specs = {
        "C(.00075,None)": dict(C=0.00075, max_iter=1000, class_weight=None),
        "B(.00075,bal)": dict(C=0.00075, max_iter=1000, class_weight="balanced"),
        "B(.003,bal)": dict(C=0.003, max_iter=1000, class_weight="balanced"),
        "B(1.0,bal)": dict(C=1.0, max_iter=1000, class_weight="balanced"),
    }
    if only is not None:
        specs = {n: specs[n] for n in only}
    return {
        name: fit_calibrated_head(X[tr], y[tr], X[cal], y[cal], **kw)
        for name, kw in specs.items()
    }


def train_heads_dense(scan_store: Path, keys=KEYS_7, neg_cap: int = 2_000_000, seed: int = 0,
                      head: str = "logistic", tree_params: dict | None = None) -> dict:
    """Train heads on the dense scan TRAIN split at (near-)true imbalance.

    The key experiment: instead of the curated 3:1 set, train on the *genome-scale* negative
    distribution the caller actually faces. Trains on **cognate** codons only (AUG +
    near_cognate) — non-cognate stays eval-only for the grounding control — keeping all train
    positives plus a capped negative sample (far more negative diversity, and a far higher
    train imbalance, than 3:1). Calibrates on the dense val split.
    """
    emb = scan_store / "embeddings"
    sm = pd.read_parquet(scan_store / "manifest.parquet")
    y = sm.label_tis.values
    cog = np.isin(sm.codon_class.values, ["AUG", "near_cognate"])
    rng = np.random.default_rng(seed)

    tr = sm.split.values == "train"
    pos = np.where(tr & cog & (y == 1))[0]
    neg = np.where(tr & cog & (y == 0))[0]
    if len(neg) > neg_cap:
        neg = rng.choice(neg, neg_cap, replace=False)
    tr_idx = np.sort(np.concatenate([pos, neg]))  # sorted -> sequential mmap reads (NFS-friendly)
    va = np.where((sm.split.values == "val") & cog)[0]
    if len(va) > 300_000:
        va = np.sort(rng.choice(va, 300_000, replace=False))
    print(f"dense train: pos={len(pos):,} neg={len(neg):,} ({len(neg) / len(pos):.0f}:1)  "
          f"cal(val)={len(va):,}", flush=True)

    Xtr = _load_full(emb, keys, tr_idx)
    Xva = _load_full(emb, keys, va)
    ytr, yva = y[tr_idx], y[va]

    if head == "logistic":
        specs = {
            "Dense(bal)": dict(C=0.00075, max_iter=1000, class_weight="balanced"),
            "Dense(None)": dict(C=0.00075, max_iter=1000, class_weight=None),
        }
        return {n: fit_calibrated_head(Xtr, ytr, Xva, yva, **kw) for n, kw in specs.items()}

    tp = dict(tree_params or {})
    if head == "xgboost":
        return {"Dense(xgb)": fit_xgb_head(Xtr, ytr, Xva, yva, seed=seed, **tp)}
    if head == "lightgbm":
        return {"Dense(lgb)": fit_lgb_head(Xtr, ytr, Xva, yva, seed=seed, **tp)}
    if head == "rf":
        return {"Dense(rf)": fit_rf_head(Xtr, ytr, Xva, yva, seed=seed, **tp)}
    raise ValueError(f"unknown head: {head!r}")


BUDGETS = (1.0, 2.0, 5.0, 10.0, 20.0)


def evaluate(scan_store: Path, heads: dict, keys=KEYS_7, budget: float = 1.0,
             save_preds: str | None = None) -> None:
    """Score heads on the dense scan TEST split at true imbalance; print a diagnostic table.

    Beyond the single ≤``budget`` operating point, reports the recall-vs-FP-budget curve and
    AUPRC so we can tell a genuinely weak ranker from a brutal operating point. Optionally
    saves all per-head predictions so any further metric is instant (no re-scoring).
    """
    emb = scan_store / "embeddings"
    sm = pd.read_parquet(scan_store / "manifest.parquet")
    te = np.where(sm.split.values == "test")[0]
    cls = sm.codon_class.values[te]
    y = sm.label_tis.values[te]
    tx = sm.transcript_id.values[te]
    cognate = np.isin(cls, ["AUG", "near_cognate"])
    noncog = cls == "non_cognate"
    npos = int((y[cognate] == 1).sum())
    ratio = (y[cognate] == 0).sum() / max(1, npos)

    print(f"\nscan TEST: {len(te):,} codons  cognate={int(cognate.sum()):,}  "
          f"non-cognate={int(noncog.sum()):,}  positives={npos:,}")
    print(f"true imbalance (neg:pos over cognate) = {ratio:.1f}:1   (curated was 3:1)\n")

    preds = _predict_chunked(emb, keys, heads, te)

    if save_preds:
        np.savez(save_preds, y=y, cognate=cognate, noncog=noncog,
                 tx=tx.astype(str), **{f"p::{n}": p for n, p in preds.items()})
        print(f"saved predictions -> {save_preds}\n")

    yc, txc = y[cognate], tx[cognate]
    base_ap = yc.mean()
    hdr = f"{'metric':<30}" + "".join(f"{n:>18}" for n in heads)
    print(hdr); print("-" * len(hdr))

    def line(label, vals, fmt="{:.3f}"):
        print(f"{label:<30}" + "".join(f"{fmt.format(v):>18}" for v in vals))

    # AUPRC — pure ranking quality at true imbalance (baseline = positive rate)
    line(f"AUPRC (base {base_ap:.4f})", [average_precision_score(yc, preds[n][cognate]) for n in heads], "{:.4f}")
    # recall-vs-budget curve — is ≤1 FP/tx just brutal, or is recall flat everywhere?
    for b in BUDGETS:
        line(f"recall @ ≤{b:g} FP/tx",
             [recall_at_fp_budget(preds[n][cognate], yc, txc, budget=b)["recall"] for n in heads])
    # grounding + calibration at the ≤1 operating threshold
    thr = {n: recall_at_fp_budget(preds[n][cognate], yc, txc, budget=1.0)["threshold"] for n in heads}
    line("non-cognate mean p (→0)", [grounding_stats(preds[n][noncog], thr[n])["mean_p"] for n in heads], "{:.4f}")
    line("non-cognate p95", [grounding_stats(preds[n][noncog], thr[n])["p95"] for n in heads], "{:.4f}")
    line("cognate Brier", [reliability(preds[n][cognate], yc)["brier"] for n in heads], "{:.4f}")
    print("\nReads: flat recall across budgets => ranking-limited; steep climb => ≤1 FP/tx is just strict.")


def score_demo(demo_store: Path, heads: dict, keys=KEYS_7) -> None:
    """Score every cognate codon of the out-of-sample demo genes with both heads."""
    emb = demo_store / "embeddings"
    m = pd.read_parquet(demo_store / "manifest.parquet")
    X = _load_full(emb, keys)  # demo store is small — load fully
    preds = {name: head["predict"](X) for name, head in heads.items()}

    gene_col = "gene" if "gene" in m.columns else "transcript_id"
    region_col = next((c for c in ("region", "region_class") if c in m.columns), None)
    print("\n=== 3-gene demo (out-of-sample) ===")
    cols = f"{'gene':<10}{'pos':>10} {'codon':>6} {'class':>13}"
    if region_col:
        cols += f" {'region':>8}"
    cols += "".join(f"{n:>26}" for n in heads)
    for g in m[gene_col].unique():
        gi = np.where(m[gene_col].values == g)[0]
        order = gi[np.argsort(m.gstart.values[gi])]
        print(f"\n{g}:")
        print(cols)
        for i in order:
            row = (f"{str(m[gene_col].values[i]):<10}{int(m.gstart.values[i]):>10} "
                   f"{str(m.codon.values[i]):>6} {str(m.codon_class.values[i]):>13}")
            if region_col:
                row += f" {str(m[region_col].values[i]):>8}"
            row += "".join(f"{preds[n][i]:>26.4f}" for n in heads)
            print(row)


def main() -> None:
    """Train Config C + Option B on the curated store, evaluate on the dense scan."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--curated-store", default="data/store")
    ap.add_argument("--scan-store", default="data/scan_store_allsplits")
    ap.add_argument("--demo-store", default=None, help="optional 3-gene demo store")
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="ag7",
                    help="ag7 = full AG+Evo2+Kozak winner; ag = AG+Kozak (pre-evo2 substrate)")
    ap.add_argument("--train", choices=["curated", "dense"], default="curated",
                    help="curated = 3:1 set (old); dense = scan TRAIN split at true imbalance")
    ap.add_argument("--neg-cap", type=int, default=2_000_000, help="dense train negative cap")
    ap.add_argument("--head", choices=["logistic", "xgboost", "lightgbm", "rf"],
                    default="logistic", help="dense-train classifier (logistic = §7 baseline)")
    ap.add_argument("--xgb-depth", type=int, default=6)
    ap.add_argument("--xgb-lr", type=float, default=0.05)
    ap.add_argument("--xgb-estimators", type=int, default=500)
    ap.add_argument("--xgb-colsample", type=float, default=0.5)
    ap.add_argument("--xgb-spw", type=float, default=None, help="scale_pos_weight; None=auto")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--save-preds", default=None, help="npz path to dump per-head test predictions")
    args = ap.parse_args()

    keys = FEATURE_SETS[args.features]
    print(f"feature set: {args.features} ({len(keys)} keys)  train={args.train}")
    if args.train == "dense":
        tree_params = None
        if args.head in ("xgboost", "lightgbm"):
            tree_params = dict(max_depth=args.xgb_depth, learning_rate=args.xgb_lr,
                               n_estimators=args.xgb_estimators,
                               colsample_bytree=args.xgb_colsample,
                               scale_pos_weight=args.xgb_spw)
        heads = {}
        if args.head == "logistic":
            heads["curated-C"] = train_heads(Path(args.curated_store), keys=keys,
                                             only=["C(.00075,None)"])["C(.00075,None)"]
        heads.update(train_heads_dense(Path(args.scan_store), keys=keys, neg_cap=args.neg_cap,
                                       head=args.head, tree_params=tree_params))
    else:
        heads = train_heads(Path(args.curated_store), keys=keys)
    evaluate(Path(args.scan_store), heads, keys=keys, budget=args.budget, save_preds=args.save_preds)
    if args.demo_store:
        score_demo(Path(args.demo_store), heads, keys=keys)


if __name__ == "__main__":
    main()
