"""Continuous initiation-efficiency regression head (Phase 4).

Regress ``log1p(max_norm_HeLa)`` on the frozen 7-key embeddings (Ridge), then test whether
ranking candidates by *predicted efficiency* recovers a better TIS ranking than the binary head.

CAVEAT (FINDINGS §5): efficiency labels exist only on the curated store, so this head trains on
curated and is evaluated on the dense TEST split — the train-curated/eval-dense regime that
collapses. A weak dense ranking here is confounded by distribution shift, NOT evidence that
efficiency signal is absent. Report accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from tisiago.caller import recall_at_fp_budget
from tisiago.dense_caller import FEATURE_SETS, _load_full, _predict_chunked
from tisiago.manifest import SITE_KEY, unique_site_indices


def fit_efficiency_head(X_train, y_train, X_val, y_val, *, alpha=1.0) -> dict:
    """Standardize, fit Ridge on log1p efficiency, report r2/rmse/spearman on val."""
    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(X_train), y_train)
    pred_val = model.predict(scaler.transform(X_val))
    metrics = {
        "r2_val": float(r2_score(y_val, pred_val)),
        "rmse_val": float(mean_squared_error(y_val, pred_val) ** 0.5),
        "spearman_val": float(spearmanr(y_val, pred_val).correlation),
    }
    linear_coef = (model.coef_ / scaler.scale_).astype(np.float32)
    linear_intercept = float(model.intercept_ - scaler.mean_ @ linear_coef)

    def predict(X):
        return np.asarray(X) @ linear_coef + linear_intercept

    def identity(values):
        return np.asarray(values)

    return {
        "predict": predict,
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "linear_coef": linear_coef,
        "linear_intercept": linear_intercept,
        "predict_from_logit": identity,
        "calibrate_raw": identity,
    }


def evaluate_as_classifier(efficiency_preds, y_binary, tx, budgets=(1.0, 5.0, 20.0)) -> dict:
    """Min-max normalize continuous preds, then score as a TIS ranker (AUPRC + recall@FP/tx)."""
    p = np.asarray(efficiency_preds, dtype=np.float64)
    p_min, p_max = p.min(), p.max()
    p_norm = (p - p_min) / (p_max - p_min + 1e-10)
    results = {"AUPRC": float(average_precision_score(y_binary, p_norm))}
    for b in budgets:
        results[f"recall@{b}FP"] = float(
            recall_at_fp_budget(p_norm, y_binary, tx, budget=b)["recall"]
        )
    return results


def _build_target(m: pd.DataFrame, col: str) -> np.ndarray:
    """Build an expression-aware log1p efficiency target.

    A missing Ribo-seq value is treated as zero only when the host gene was assayed
    and marked expressed. Unexpressed/unmeasured rows remain NaN and are excluded
    from regression. Duplicate per-cell-line call rows are aggregated by maximum
    observed efficiency for the same transcript-relative site.
    """
    if col not in m:
        raise ValueError(f"target column not found: {col}")
    suffix = col.removeprefix("max_norm_")
    expressed_col = f"expressed_{suffix}"
    if expressed_col not in m:
        raise ValueError(f"expression column not found: {expressed_col}")

    groups = m.groupby(list(SITE_KEY), sort=False, dropna=False)
    measured = groups[col].transform("max").to_numpy(dtype=np.float64)
    expressed = groups[expressed_col].transform("max").to_numpy(dtype=bool)
    eligible = np.isfinite(measured) | expressed
    y = np.full(len(m), np.nan, dtype=np.float64)
    y[eligible] = 0.0
    has_value = np.isfinite(measured)
    y[has_value] = np.log1p(measured[has_value])
    return y


def main() -> None:
    """Alpha-sweep Ridge on curated efficiency, then eval as classifier on the dense TEST split."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--curated-store", default="data/store")
    ap.add_argument("--scan-store", default="data/dense_ag7")
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="ag7")
    ap.add_argument("--target", default="max_norm_HeLa")
    ap.add_argument("--alphas", default="0.01,0.1,1.0,10.0,100.0,1000.0")
    ap.add_argument(
        "--save-preds", default=None, help="npz to dump dense-TEST preds for the best alpha"
    )
    args = ap.parse_args()

    keys = FEATURE_SETS[args.features]
    cur = Path(args.curated_store)
    m = pd.read_parquet(cur / "manifest.parquet")
    y_full = _build_target(m, args.target)
    unique = unique_site_indices(m)
    eligible = np.zeros(len(m), dtype=bool)
    eligible[unique] = np.isfinite(y_full[unique])
    tr = (m.split.values == "train") & eligible
    va = (m.split.values == "val") & eligible
    X = _load_full(cur / "embeddings", keys)
    print(f"curated: train={int(tr.sum())} val={int(va.sum())} dim={X.shape[1]} "
          f"target={args.target} (nonzero={int((y_full > 0).sum())})", flush=True)

    best = None
    for alpha in [float(a) for a in args.alphas.split(",")]:
        head = fit_efficiency_head(X[tr], y_full[tr], X[va], y_full[va], alpha=alpha)
        mt = head["metrics"]
        print(f"alpha={alpha:>8.2f}  R2={mt['r2_val']:.4f}  rho={mt['spearman_val']:.4f}  "
              f"rmse={mt['rmse_val']:.4f}", flush=True)
        if best is None or mt["spearman_val"] > best[1]["metrics"]["spearman_val"]:
            best = (alpha, head)

    alpha, head = best
    print(
        f"\nbest alpha={alpha} (by val spearman). Scoring dense TEST as a classifier...", flush=True
    )

    sm = pd.read_parquet(Path(args.scan_store) / "manifest.parquet")
    te = np.where(sm.split.values == "test")[0]
    cog = np.isin(sm.codon_class.values[te], ["AUG", "near_cognate"])
    preds = _predict_chunked(
        Path(args.scan_store) / "embeddings", keys, {"ridge": head}, te
    )["ridge"]
    yc = sm.label_tis.values[te][cog]
    txc = sm.transcript_id.values[te][cog]
    out = evaluate_as_classifier(preds[cog], yc, txc, budgets=(1.0, 5.0, 20.0))
    print("\ndense TEST (regression-as-classifier):")
    for k, v in out.items():
        print(f"  {k:<14} {v:.4f}")
    print("\nCAVEAT: trained curated / evaluated dense (FINDINGS §5 shift) — interpret with care.")

    if args.save_preds:
        np.savez(args.save_preds, y=sm.label_tis.values[te], cognate=cog,
                 tx=sm.transcript_id.values[te].astype(str),
                 **{"p::Ridge(eff)": preds})
        print(f"saved -> {args.save_preds}")


if __name__ == "__main__":
    main()
