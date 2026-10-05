# Phase 3-4 Head Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether a richer classifier (gradient-boosted trees) or a continuous
efficiency-regression head beats the dense-trained logistic head's **0.300 recall @ ≤1 FP/tx**
ceiling on the same 7-key (19,620-dim) features — or establish that 0.300 is the *feature*
ceiling and the next lever is better features, not better heads.

**Architecture:** All new heads conform to the existing `dense_caller.evaluate()` contract — a
head is a `dict` with a `"predict"` callable mapping `[N, D] float32 → [N] probabilities`. New
tree heads live in `head_xgb.py`; the regression head in `efficiency_head.py`. `dense_caller.py`
gains a `--head` switch that selects which classifier `train_heads_dense` fits. Every head saves
its dense-TEST predictions to `.npz`, and a final `compare_heads.py` assembles the headline table
from those `.npz` files — so the heavy SLURM scoring runs once and any later metric is instant.

**Tech Stack:** Python 3.11, numpy, pandas, scikit-learn (StandardScaler, Ridge, IsotonicRegression),
xgboost, lightgbm. CPU only — no GPU, no new embedding extraction. pytest for unit tests.

## Global Constraints

- **Train on dense, always.** Classification heads train on the dense scan TRAIN split at true
  imbalance (~49:1 cognate), calibrate on dense VAL, evaluate on dense TEST at ~230:1. Never the
  curated 3:1 set. The one exception is the Phase-4 regression *target*, which only exists on the
  curated store (efficiency labels) — see the Phase-4 caveat below.
- **Same evaluation protocol for every head.** All heads scored on the same dense TEST split with
  the same `tisiago.caller.recall_at_fp_budget` and `tisiago.scan_eval.grounding_stats`. Cognate =
  `codon_class ∈ {AUG, near_cognate}`; non-cognate (`non_cognate`) is eval-only grounding, never
  trained on.
- **Calibration = manual isotonic**, mirroring `caller.fit_calibrated_head`: build `predict_raw`,
  fit `IsotonicRegression(out_of_bounds="clip")` on the calibration (dense VAL) set, wrap into
  `predict`. Do **not** use `CalibratedClassifierCV(cv="prefit")` (deprecated/breaking on newer
  sklearn).
- **Head contract:** every fit function returns a dict containing at least
  `{"predict": callable, "model": <fitted>, "scaler": <StandardScaler|None>}`. `predict` takes a
  `[N, D] float32` array and returns a length-`N` array of probabilities in `[0, 1]` (regression
  head excepted — it returns a monotone score; see Task 3).
- **7-key feature stack** (`dense_caller.KEYS_7`, 19,620-dim): AG16k off0 (1536) + AG131k off0
  (1536) + Evo2 blk28 off{0,3,6,9} (4096×4) + Kozak one-hot (164).
- **Store:** default `--scan-store /lab/ops_analysis_ssd/test_matteo/tisiago_store/dense_ag7`
  (SSD, 4,329,984 rows, full `codon_class/split/label_tis` manifest — the validated §7 substrate).
  `data/dense_exp_store` is the identical NFS twin (fallback). Curated store: `data/store`.
- **Save predictions.** Every dense-TEST scoring run writes per-head preds to `.npz` via
  `--save-preds`. Reference baseline to reproduce/keep: **logistic ag7 Dense(None) = 0.300 recall
  @ ≤1 FP/tx, AUPRC 0.307, non-cog mean 0.0003** (FINDINGS §7).
- **No commits of `data/` artifacts** — `.npz` preds and stores are gitignored. Commit only code,
  tests, and docs.
- **Phase-4 distribution-shift caveat (must be reported, not hidden):** the regression head trains
  on the **curated** store (the only place efficiency labels exist) and is evaluated on the
  **dense** TEST split. This is exactly the train-curated/eval-dense setup that collapses in
  FINDINGS §5. Any Phase-4 result must be reported *with* this caveat — a weak dense ranking from
  the regression head is confounded by distribution shift and is **not** evidence that efficiency
  signal is absent.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/tisiago/head_xgb.py` | **NEW.** Tree-ensemble heads: `fit_xgb_head`, `fit_lgb_head`, `fit_rf_head`. Each fits, manual-isotonic-calibrates on val, returns the head dict. |
| `src/tisiago/dense_caller.py` | **MODIFY.** Add `--head {logistic,xgboost,lightgbm,rf}` + tree hyperparam args; route `train_heads_dense` to the chosen fit function. Logistic path unchanged. |
| `src/tisiago/efficiency_head.py` | **NEW.** `fit_efficiency_head` (Ridge on `log1p(max_norm_HeLa)`), `evaluate_as_classifier`, and a `main()` that does the alpha sweep + dense-TEST classifier eval. |
| `scripts/compare_heads.py` | **NEW.** Load every `data/preds_*.npz`, recompute the headline metrics per head, print + write the comparison table. |
| `tests/test_head_xgb.py` | **NEW.** Predict-dict contract on synthetic data. |
| `tests/test_efficiency_head.py` | **NEW.** Ridge metrics + `evaluate_as_classifier` normalization. |
| `tests/test_compare_heads.py` | **NEW.** Table assembly from synthetic `.npz`. |
| `FINDINGS.md` | **MODIFY (reporting task).** Add §9 comparison table + interpretation. |
| `ROADMAP.md` | **MODIFY (reporting task).** Mark P3 head-exploration outcome. |

Out of scope (YAGNI, deferred): Phase 3c attention-pooling over Evo2 offsets (torch). The spec
gates it on "only if 3a shows nonlinear heads help" — it is a conditional follow-up, not part of
this plan.

---

## Task 1: Tree-ensemble head module (`head_xgb.py`)

**Files:**
- Create: `src/tisiago/head_xgb.py`
- Test: `tests/test_head_xgb.py`

**Interfaces:**
- Consumes: nothing from this repo (pure sklearn/xgboost/lightgbm). Mirrors the head-dict shape of
  `caller.fit_calibrated_head` → `{"predict", "predict_raw", "model", "scaler"}`.
- Produces:
  - `fit_xgb_head(X_train, y_train, X_val, y_val, *, n_estimators=500, max_depth=6, learning_rate=0.05, scale_pos_weight=None, subsample=0.8, colsample_bytree=0.5, seed=0) -> dict`
  - `fit_lgb_head(X_train, y_train, X_val, y_val, *, n_estimators=500, max_depth=6, learning_rate=0.05, scale_pos_weight=None, subsample=0.8, colsample_bytree=0.5, seed=0) -> dict`
  - `fit_rf_head(X_train, y_train, X_val, y_val, *, n_estimators=500, max_depth=12, seed=0) -> dict`
  - Each returns `{"predict": callable, "predict_raw": callable, "model": fitted, "scaler": None}`.
    `predict(X)` returns isotonic-calibrated probabilities in `[0, 1]`, shape `[N]`.

**Dependency install (do this first, once):**
```bash
eval "$(conda shell.bash hook)" && conda activate tisiago
python -c "import xgboost" 2>/dev/null || pip install xgboost
python -c "import lightgbm" 2>/dev/null || pip install lightgbm
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_head_xgb.py
"""Contract tests for tree-ensemble TIS heads: predict() returns calibrated [0,1] probs."""
import numpy as np
import pytest

from tisiago.head_xgb import fit_xgb_head, fit_lgb_head, fit_rf_head


def _toy(n=400, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    # a learnable signal: positive when first two coords sum high, plus class imbalance
    logit = X[:, 0] + X[:, 1]
    y = (logit + rng.standard_normal(n) * 0.5 > 1.0).astype(int)
    return X, y


@pytest.mark.parametrize("fit", [fit_xgb_head, fit_lgb_head, fit_rf_head])
def test_head_contract(fit):
    Xtr, ytr = _toy(seed=0)
    Xva, yva = _toy(seed=1)
    head = fit(Xtr, ytr, Xva, yva, n_estimators=40, seed=0)
    assert callable(head["predict"])
    assert head["scaler"] is None
    p = head["predict"](Xva)
    assert p.shape == (Xva.shape[0],)
    assert p.min() >= 0.0 and p.max() <= 1.0
    # learns *something*: positives rank above negatives on average
    assert p[yva == 1].mean() > p[yva == 0].mean()


def test_xgb_scale_pos_weight_auto():
    # imbalanced toy: auto scale_pos_weight should be ~ neg/pos, head still calibrates to [0,1]
    Xtr, ytr = _toy(n=800, seed=2)
    Xva, yva = _toy(n=400, seed=3)
    head = fit_xgb_head(Xtr, ytr, Xva, yva, n_estimators=40, scale_pos_weight=None, seed=0)
    p = head["predict"](Xva)
    assert np.isfinite(p).all()
    assert 0.0 <= p.min() and p.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `eval "$(conda shell.bash hook)" && conda activate tisiago && python -m pytest tests/test_head_xgb.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tisiago.head_xgb'` (after deps installed; if xgboost/lightgbm missing, install per the block above first).

- [ ] **Step 3: Write minimal implementation**

```python
# src/tisiago/head_xgb.py
"""Gradient-boosted tree (and random-forest) heads for dense TIS classification.

The dense-trained logistic head (FINDINGS §7) tops out at 0.300 recall @ ≤1 FP/tx. Trees can
capture interactions between the AG regional embedding and the Evo2 nucleotide embedding that a
linear boundary misses, and they need no feature scaling. Each fit mirrors
``caller.fit_calibrated_head``: train, then isotonic-calibrate on the held-out val split, so the
returned ``predict`` plugs straight into ``dense_caller.evaluate`` with probabilities in [0, 1].
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression


def _auto_spw(y_train) -> float:
    """scale_pos_weight = negatives / positives (XGBoost/LightGBM imbalance lever)."""
    y = np.asarray(y_train)
    return float((y == 0).sum()) / max(1, int((y == 1).sum()))


def _calibrate(predict_raw, X_val, y_val, model):
    """Isotonic-calibrate raw scores on val; return the standard head dict (scaler=None)."""
    iso = IsotonicRegression(out_of_bounds="clip").fit(predict_raw(X_val), y_val)

    def predict(X):
        return iso.predict(predict_raw(X))

    return {"predict": predict, "predict_raw": predict_raw, "model": model, "scaler": None}


def fit_xgb_head(X_train, y_train, X_val, y_val, *, n_estimators=500, max_depth=6,
                 learning_rate=0.05, scale_pos_weight=None, subsample=0.8,
                 colsample_bytree=0.5, seed=0) -> dict:
    """Train an XGBoost classifier (hist), early-stop on val, isotonic-calibrate on val."""
    import xgboost as xgb

    if scale_pos_weight is None:
        scale_pos_weight = _auto_spw(y_train)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight, subsample=subsample, colsample_bytree=colsample_bytree,
        tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50,
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def predict_raw(X):
        return clf.predict_proba(X)[:, 1]

    return _calibrate(predict_raw, X_val, y_val, clf)


def fit_lgb_head(X_train, y_train, X_val, y_val, *, n_estimators=500, max_depth=6,
                 learning_rate=0.05, scale_pos_weight=None, subsample=0.8,
                 colsample_bytree=0.5, seed=0) -> dict:
    """Train a LightGBM classifier, isotonic-calibrate on val."""
    import lightgbm as lgb

    if scale_pos_weight is None:
        scale_pos_weight = _auto_spw(y_train)
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight, subsample=subsample, colsample_bytree=colsample_bytree,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_train, y_train)

    def predict_raw(X):
        return clf.predict_proba(X)[:, 1]

    return _calibrate(predict_raw, X_val, y_val, clf)


def fit_rf_head(X_train, y_train, X_val, y_val, *, n_estimators=500, max_depth=12, seed=0) -> dict:
    """Train a balanced random forest, isotonic-calibrate on val.

    Diagnostic only: RF at the full 2M×19.6k scale is memory-prohibitive — fit it on a train
    subsample (see the runbook). If RF beats logistic but XGBoost doesn't, the gain is ensemble
    averaging, not interaction learning.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    def predict_raw(X):
        return clf.predict_proba(X)[:, 1]

    return _calibrate(predict_raw, X_val, y_val, clf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_head_xgb.py -v`
Expected: PASS (4 cases: 3 parametrized contract + the spw-auto case).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/tisiago/head_xgb.py tests/test_head_xgb.py
git add src/tisiago/head_xgb.py tests/test_head_xgb.py
git commit -m "feat(head): tree-ensemble heads (xgb/lgb/rf) with isotonic calibration"
```

---

## Task 2: `--head` integration into `dense_caller.py`

**Files:**
- Modify: `src/tisiago/dense_caller.py` (`train_heads_dense`, `main`, argparse)
- Test: `tests/test_dense_caller_head.py` (new)

**Interfaces:**
- Consumes: `tisiago.head_xgb.fit_xgb_head`, `fit_lgb_head`, `fit_rf_head` (Task 1);
  existing `train_heads_dense` data-prep, `fit_calibrated_head`, `_load_full`.
- Produces:
  - `train_heads_dense(scan_store, keys=KEYS_7, neg_cap=2_000_000, seed=0, head="logistic", tree_params=None) -> dict`.
    For `head="logistic"`: unchanged — returns `{"Dense(bal)", "Dense(None)"}`. For a tree head:
    returns `{f"Dense({tag})": <head dict>}` where `tag ∈ {xgb, lgb, rf}`.
  - CLI: `--head {logistic,xgboost,lightgbm,rf}` (default `logistic`), plus
    `--xgb-depth`, `--xgb-lr`, `--xgb-estimators`, `--xgb-colsample`, `--xgb-spw` (used by
    xgboost/lightgbm). RF uses its own defaults.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dense_caller_head.py
"""train_heads_dense routes to the selected classifier; logistic path is unchanged."""
import numpy as np
import pandas as pd
import pytest

import tisiago.dense_caller as dc


@pytest.fixture
def tiny_store(tmp_path):
    """A minimal scan store: 2 keys, train+val+test rows, cognate+non_cognate classes."""
    n = 600
    rng = np.random.default_rng(0)
    # two tiny keys under embeddings/a/ so keys=["a/x.npy","a/z.npy"] resolve
    (tmp_path / "embeddings" / "a").mkdir(parents=True)
    np.save(tmp_path / "embeddings" / "a" / "x.npy", rng.standard_normal((n, 4)).astype(np.float16))
    np.save(tmp_path / "embeddings" / "a" / "z.npy", rng.standard_normal((n, 3)).astype(np.float16))
    split = np.array(["train"] * 300 + ["val"] * 150 + ["test"] * 150)
    klass = rng.choice(["AUG", "near_cognate", "non_cognate"], n, p=[0.1, 0.6, 0.3])
    y = ((klass != "non_cognate") & (rng.random(n) < 0.2)).astype(int)
    m = pd.DataFrame({
        "row_idx": np.arange(n), "transcript_id": rng.integers(0, 20, n),
        "codon_class": klass, "split": split, "label_tis": y,
    })
    m.to_parquet(tmp_path / "manifest.parquet")
    return tmp_path


def test_logistic_unchanged(tiny_store):
    keys = ["a/x.npy", "a/z.npy"]
    heads = dc.train_heads_dense(tiny_store, keys=keys, neg_cap=1000, head="logistic")
    assert set(heads) == {"Dense(bal)", "Dense(None)"}


def test_xgb_route(tiny_store):
    keys = ["a/x.npy", "a/z.npy"]
    heads = dc.train_heads_dense(tiny_store, keys=keys, neg_cap=1000, head="xgboost",
                                 tree_params={"n_estimators": 20})
    assert set(heads) == {"Dense(xgb)"}
    p = heads["Dense(xgb)"]["predict"](np.zeros((5, 7), dtype=np.float32))
    assert p.shape == (5,) and p.min() >= 0.0 and p.max() <= 1.0


def test_unknown_head_raises(tiny_store):
    with pytest.raises((ValueError, KeyError)):
        dc.train_heads_dense(tiny_store, keys=["a/x.npy", "a/z.npy"], head="banana")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dense_caller_head.py -v`
Expected: FAIL — `train_heads_dense() got an unexpected keyword argument 'head'`.

- [ ] **Step 3: Write minimal implementation**

In `src/tisiago/dense_caller.py`, add the import near the top imports:

```python
from tisiago.head_xgb import fit_lgb_head, fit_rf_head, fit_xgb_head
```

Replace the `train_heads_dense` head-fitting tail (the `specs = {...}` / `return {...}` block,
currently lines ~147-151) with a head router. The data-prep above it (`tr_idx`, `va`, `Xtr`,
`Xva`) is unchanged:

```python
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
```

Update the `train_heads_dense` signature line to:

```python
def train_heads_dense(scan_store: Path, keys=KEYS_7, neg_cap: int = 2_000_000, seed: int = 0,
                      head: str = "logistic", tree_params: dict | None = None) -> dict:
```

In `main()`, add CLI args after the existing `--neg-cap` line:

```python
    ap.add_argument("--head", choices=["logistic", "xgboost", "lightgbm", "rf"],
                    default="logistic", help="dense-train classifier (logistic = §7 baseline)")
    ap.add_argument("--xgb-depth", type=int, default=6)
    ap.add_argument("--xgb-lr", type=float, default=0.05)
    ap.add_argument("--xgb-estimators", type=int, default=500)
    ap.add_argument("--xgb-colsample", type=float, default=0.5)
    ap.add_argument("--xgb-spw", type=float, default=None, help="scale_pos_weight; None=auto")
```

Replace the dense-mode head construction in `main()` (the `if args.train == "dense":` block,
currently lines ~256-260) with:

```python
    if args.train == "dense":
        tree_params = None
        if args.head in ("xgboost", "lightgbm"):
            tree_params = dict(max_depth=args.xgb_depth, learning_rate=args.xgb_lr,
                               n_estimators=args.xgb_estimators, colsample_bytree=args.xgb_colsample,
                               scale_pos_weight=args.xgb_spw)
        heads = {}
        if args.head == "logistic":
            heads["curated-C"] = train_heads(Path(args.curated_store), keys=keys,
                                             only=["C(.00075,None)"])["C(.00075,None)"]
        heads.update(train_heads_dense(Path(args.scan_store), keys=keys, neg_cap=args.neg_cap,
                                       head=args.head, tree_params=tree_params))
    else:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dense_caller_head.py -v`
Expected: PASS (3 cases). Then confirm no regression: `python -m pytest tests/ -q`.

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/tisiago/dense_caller.py tests/test_dense_caller_head.py
git add src/tisiago/dense_caller.py tests/test_dense_caller_head.py
git commit -m "feat(dense_caller): --head switch routes dense training to tree heads"
```

---

## Task 3: Efficiency regression head (`efficiency_head.py`)

**Files:**
- Create: `src/tisiago/efficiency_head.py`
- Test: `tests/test_efficiency_head.py`

**Interfaces:**
- Consumes: `dense_caller._load_full`, `dense_caller._predict_chunked`, `dense_caller.FEATURE_SETS`,
  `caller.recall_at_fp_budget`.
- Produces:
  - `fit_efficiency_head(X_train, y_train, X_val, y_val, *, alpha=1.0) -> dict` with
    `{"predict", "model", "scaler", "metrics"}`; `metrics` has `r2_val`, `rmse_val`, `spearman_val`.
  - `evaluate_as_classifier(efficiency_preds, y_binary, tx, budgets=(1.0, 5.0, 20.0)) -> dict`
    with `"AUPRC"` and `f"recall@{b}FP"` keys; min-max-normalizes preds to [0,1] (div-zero safe).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_efficiency_head.py
"""Ridge efficiency head: metrics dict + classifier-thresholding of continuous predictions."""
import numpy as np

from tisiago.efficiency_head import evaluate_as_classifier, fit_efficiency_head


def test_fit_returns_metrics_and_predict():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8)).astype(np.float32)
    w = rng.standard_normal(8)
    y = X @ w + rng.standard_normal(300) * 0.1
    head = fit_efficiency_head(X[:200], y[:200], X[200:], y[200:], alpha=1.0)
    assert set(["r2_val", "rmse_val", "spearman_val"]).issubset(head["metrics"])
    assert head["metrics"]["r2_val"] > 0.5  # recoverable linear signal
    p = head["predict"](X[:5])
    assert p.shape == (5,)


def test_evaluate_as_classifier_keys_and_norm():
    rng = np.random.default_rng(1)
    preds = rng.standard_normal(200) * 5.0  # arbitrary scale, not [0,1]
    y = (preds > preds.mean()).astype(int)
    tx = rng.integers(0, 10, 200)
    out = evaluate_as_classifier(preds, y, tx, budgets=(1.0, 5.0))
    assert "AUPRC" in out and "recall@1.0FP" in out and "recall@5.0FP" in out
    assert 0.0 <= out["AUPRC"] <= 1.0


def test_evaluate_as_classifier_constant_preds_safe():
    # all-equal predictions must not divide by zero
    preds = np.full(50, 3.0)
    y = np.zeros(50, dtype=int); y[:5] = 1
    tx = np.arange(50) % 7
    out = evaluate_as_classifier(preds, y, tx, budgets=(1.0,))
    assert np.isfinite(out["AUPRC"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_efficiency_head.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tisiago.efficiency_head'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tisiago/efficiency_head.py
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

    def predict(X):
        return model.predict(scaler.transform(X))

    return {"predict": predict, "model": model, "scaler": scaler, "metrics": metrics}


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
    """log1p(efficiency) where measured, else 0 (no measured initiation)."""
    y = np.zeros(len(m), dtype=np.float64)
    has = m[col].notna().values
    y[has] = np.log1p(m.loc[has, col].values)
    return y


def main() -> None:
    """Alpha-sweep Ridge on curated efficiency, then eval as classifier on the dense TEST split."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--curated-store", default="data/store")
    ap.add_argument("--scan-store",
                    default="/lab/ops_analysis_ssd/test_matteo/tisiago_store/dense_ag7")
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="ag7")
    ap.add_argument("--target", default="max_norm_HeLa")
    ap.add_argument("--alphas", default="0.01,0.1,1.0,10.0,100.0,1000.0")
    ap.add_argument("--save-preds", default=None, help="npz to dump dense-TEST preds for the best alpha")
    args = ap.parse_args()

    keys = FEATURE_SETS[args.features]
    cur = Path(args.curated_store)
    m = pd.read_parquet(cur / "manifest.parquet")
    y_full = _build_target(m, args.target)
    tr = m.split.values == "train"
    va = m.split.values == "val"
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
    print(f"\nbest alpha={alpha} (by val spearman). Scoring dense TEST as a classifier...", flush=True)

    sm = pd.read_parquet(Path(args.scan_store) / "manifest.parquet")
    te = np.where(sm.split.values == "test")[0]
    cog = np.isin(sm.codon_class.values[te], ["AUG", "near_cognate"])
    preds = _predict_chunked(Path(args.scan_store) / "embeddings", keys,
                             {"ridge": {"predict": head["predict"]}}, te)["ridge"]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_efficiency_head.py -v`
Expected: PASS (3 cases).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/tisiago/efficiency_head.py tests/test_efficiency_head.py
git add src/tisiago/efficiency_head.py tests/test_efficiency_head.py
git commit -m "feat(phase4): Ridge efficiency regression head + classifier eval"
```

---

## Task 4: Head-comparison table builder (`scripts/compare_heads.py`)

**Files:**
- Create: `scripts/compare_heads.py`
- Test: `tests/test_compare_heads.py`

**Interfaces:**
- Consumes: `caller.recall_at_fp_budget`, `caller.reliability`, `scan_eval.grounding_stats`,
  and the `.npz` prediction files written by `dense_caller.evaluate(--save-preds)` /
  `efficiency_head.main(--save-preds)`. Each `.npz` has arrays `y`, `cognate`, `tx`, and one or
  more `p::<headname>` arrays; `noncog` is present for classifier preds (absent for the regression
  npz — handle its absence: grounding columns blank).
- Produces: `metrics_for(npz_path) -> list[dict]` (one dict per `p::` head in the file) and a
  `main()` that prints + writes `data/head_comparison.tsv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare_heads.py
"""compare_heads recomputes the headline metrics per head from saved .npz prediction files."""
import numpy as np

from scripts.compare_heads import metrics_for


def _write_npz(path, with_noncog=True, seed=0):
    rng = np.random.default_rng(seed)
    n = 2000
    klass_cog = rng.random(n) < 0.7
    y = (klass_cog & (rng.random(n) < 0.1)).astype(int)
    tx = rng.integers(0, 50, n)
    # a head that ranks positives a bit higher
    p = rng.random(n) * 0.3 + y * 0.4
    d = {"y": y, "cognate": klass_cog, "tx": tx.astype(str), "p::HeadA": p}
    if with_noncog:
        d["noncog"] = ~klass_cog
    np.savez(path, **d)


def test_metrics_for_classifier_npz(tmp_path):
    f = tmp_path / "preds_a.npz"
    _write_npz(f, with_noncog=True)
    rows = metrics_for(f)
    assert len(rows) == 1
    r = rows[0]
    assert r["head"] == "HeadA"
    assert 0.0 <= r["AUPRC"] <= 1.0
    assert "recall@1FP" in r and "noncog_mean_p" in r
    assert 0.0 <= r["recall@1FP"] <= 1.0


def test_metrics_for_handles_missing_noncog(tmp_path):
    f = tmp_path / "preds_reg.npz"
    _write_npz(f, with_noncog=False)
    rows = metrics_for(f)
    assert len(rows) == 1
    # grounding unavailable -> NaN, not a crash
    assert np.isnan(rows[0]["noncog_mean_p"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare_heads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.compare_heads'` (the conftest sets
`pythonpath=["."]`, so `scripts` is importable; the module just doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/compare_heads.py
"""Assemble the Phase-3/4 head-comparison table from saved dense-TEST prediction .npz files.

Each scoring run (dense_caller --save-preds, efficiency_head --save-preds) writes y/cognate/tx and
one or more ``p::<head>`` arrays. This recomputes the headline metrics identically across heads so
the comparison is apples-to-apples, and writes data/head_comparison.tsv.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from tisiago.caller import recall_at_fp_budget, reliability
from tisiago.scan_eval import grounding_stats

BUDGETS = (1.0, 5.0, 20.0)


def metrics_for(npz_path) -> list[dict]:
    """One metrics dict per ``p::`` head in a saved prediction file."""
    from sklearn.metrics import average_precision_score

    z = np.load(npz_path, allow_pickle=True)
    y = z["y"]
    cog = z["cognate"]
    tx = z["tx"]
    noncog = z["noncog"] if "noncog" in z.files else None
    yc, txc = y[cog], tx[cog]
    rows = []
    for key in [k for k in z.files if k.startswith("p::")]:
        p = z[key]
        pc = p[cog]
        row = {
            "head": key[3:],
            "source": Path(npz_path).name,
            "AUPRC": float(average_precision_score(yc, pc)),
        }
        thr = recall_at_fp_budget(pc, yc, txc, budget=1.0)["threshold"]
        for b in BUDGETS:
            row[f"recall@{int(b)}FP"] = float(
                recall_at_fp_budget(pc, yc, txc, budget=b)["recall"]
            )
        row["brier"] = float(reliability(pc, yc)["brier"])
        if noncog is not None:
            row["noncog_mean_p"] = float(grounding_stats(p[noncog], thr)["mean_p"])
        else:
            row["noncog_mean_p"] = float("nan")
        rows.append(row)
    return rows


def main() -> None:
    """Glob preds .npz, print the comparison table, write data/head_comparison.tsv."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="data/preds_*.npz")
    ap.add_argument("--out", default="data/head_comparison.tsv")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(args.glob)):
        rows.extend(metrics_for(f))
    if not rows:
        print(f"no prediction files matched {args.glob}")
        return

    cols = ["head", "AUPRC", "recall@1FP", "recall@5FP", "recall@20FP", "noncog_mean_p",
            "brier", "source"]
    hdr = "".join(f"{c:>16}" if c != "head" else f"{c:<20}" for c in cols)
    print(hdr)
    lines = ["\t".join(cols)]
    for r in sorted(rows, key=lambda d: -d["AUPRC"]):
        cells = [r["head"]] + [
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols[1:]
        ]
        print("".join(f"{c:>16}" if i else f"{c:<20}" for i, c in enumerate(cells)))
        lines.append("\t".join(cells))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_compare_heads.py -v`
Expected: PASS (2 cases). Then full suite: `python -m pytest tests/ -q`.

- [ ] **Step 5: Lint + commit**

```bash
ruff check scripts/compare_heads.py tests/test_compare_heads.py
git add scripts/compare_heads.py tests/test_compare_heads.py
git commit -m "feat(compare): head-comparison table builder from saved preds"
```

---

## Part B — Experiment runbook (controller-executed, not TDD)

These produce the numbers. Run **after** Tasks 1-4 land. Each is a CPU SLURM job on a big-mem
node. **Memory reality (correcting the spec's 192G estimate):** the dense 2M train matrix alone is
2.04M × 19,620 × 4 B ≈ **160 GB float32**; the logistic 2M run already peaked ~472 GB. Trees add a
DMatrix/quantization on top. Run on a **983 GB node** (`it-bigboy`/`c5b8`, partition 20) or the
1.47 TB `r2d22` (partition 24) with `--mem=750G`. Namespace job names with `${USER}_`. Cancel by
**job ID only**.

Look up limits first: `curl -s http://slurmstatus.wi.mit.edu/limits.html`.

```bash
cd /lab/barcheese01/mdiberna/tisiago
SSD=/lab/ops_analysis_ssd/test_matteo/tisiago_store/dense_ag7
COMMON="--train dense --features ag7 --scan-store $SSD"

# B0. Logistic baseline @ 2M — reproduce the §7 headline, save preds (reference row).
srun --partition=20 --nodelist=it-bigboy --mem=750G --cpus-per-task=16 --time=12:00:00 \
  -J ${USER}_head_logit python -m tisiago.dense_caller $COMMON --head logistic \
  --save-preds data/preds_logistic.npz

# B1. XGBoost default (depth 6, lr 0.05, colsample 0.5, auto spw).
srun --partition=20 --nodelist=it-bigboy --mem=750G --cpus-per-task=16 --time=12:00:00 \
  -J ${USER}_head_xgb python -m tisiago.dense_caller $COMMON --head xgboost \
  --save-preds data/preds_xgb.npz

# B2. LightGBM default.
srun --partition=20 --nodelist=it-bigboy --mem=750G --cpus-per-task=16 --time=12:00:00 \
  -J ${USER}_head_lgb python -m tisiago.dense_caller $COMMON --head lightgbm \
  --save-preds data/preds_lgb.npz

# B3. Ridge efficiency (Phase 4) — curated train, dense-TEST eval. Lighter (curated 192k).
srun --partition=20 --mem=192G --cpus-per-task=16 --time=4:00:00 \
  -J ${USER}_head_ridge python -m tisiago.efficiency_head --features ag7 --scan-store $SSD \
  --save-preds data/preds_ridge.npz
```

- [ ] **B0** logistic baseline → `data/preds_logistic.npz` (expect recall@1FP ≈ 0.300, AUPRC ≈ 0.307).
- [ ] **B1** XGBoost default → `data/preds_xgb.npz`. If it beats 0.300, sweep `--xgb-depth {4,6,8}`
  × `--xgb-lr {0.01,0.05,0.1}` (saving `data/preds_xgb_d{D}_lr{L}.npz`). If it ties/loses, **stop
  the sweep** — record it as evidence for the feature-ceiling conclusion.
- [ ] **B2** LightGBM default → `data/preds_lgb.npz`.
- [ ] **B3** Ridge efficiency → printed R²/ρ alpha sweep + `data/preds_ridge.npz`.
- [ ] **B4 (optional, diagnostic):** RF on a 500k subsample only — `--head rf --neg-cap 500000`.
  Skip if XGBoost already settles the interaction question. RF at full scale is memory-prohibitive.

Monitor: `squeue -u $USER`; tail the `slurm-*.out`. On OOM, drop to `--neg-cap 1000000` (peak
~300 GB) and note the smaller train set in the writeup.

---

## Part C — Reporting (Task 5)

### Task 5: Comparison table + FINDINGS §9 + ROADMAP

**Files:**
- Modify: `FINDINGS.md` (add §9), `ROADMAP.md` (P3 outcome line)
- Run: `scripts/compare_heads.py`

This task runs only after Part B preds exist. It is a controller/subagent task with the numbers in
hand — no code logic, so no new tests.

- [ ] **Step 1: Build the table**

```bash
python scripts/compare_heads.py --glob 'data/preds_*.npz' --out data/head_comparison.tsv
column -t data/head_comparison.tsv
```

- [ ] **Step 2: Write FINDINGS §9** with the table and one of the two conclusions:
  - *If any head beats 0.300 recall @ ≤1 FP/tx:* report the winner, its hyperparameters, and the
    margin; note whether the lift is ranking (AUPRC) or operating-point.
  - *If all heads plateau at ~0.30:* state explicitly that **0.300 is the feature ceiling on ag7**
    — the linear head is not the bottleneck, and the next lever is better features (Phase 5 LoRA),
    not richer classifiers. This is a first-class result.
  - Phase-4 row: report the regression-as-classifier numbers **with** the train-curated/eval-dense
    caveat from Global Constraints. Also report the in-distribution curated val R²/ρ (does the head
    predict efficiency *at all*?) separately from its dense-TEST ranking.

- [ ] **Step 3: Update ROADMAP** P3 line with the outcome (beat-the-ceiling vs feature-ceiling).

- [ ] **Step 4: Commit**

```bash
git add FINDINGS.md ROADMAP.md
git commit -m "docs(findings): §9 head-exploration comparison + Phase-4 efficiency"
```

---

## Verification

1. **Unit suite green:** `python -m pytest tests/ -q` — all prior tests plus the 4 new test files
   (`test_head_xgb`, `test_dense_caller_head`, `test_efficiency_head`, `test_compare_heads`) pass.
2. **Logistic path unmodified:** `test_logistic_unchanged` proves `--head logistic` still returns
   `{Dense(bal), Dense(None)}`; B0 reproduces recall@1FP ≈ 0.300 / AUPRC ≈ 0.307 (the §7 headline)
   — confirming the `--head` refactor didn't disturb the baseline.
3. **Tree heads obey the contract:** `test_head_contract` proves `predict` returns `[N]`
   probabilities in `[0, 1]` that rank positives above negatives.
4. **Comparison is apples-to-apples:** `compare_heads.py` recomputes every head's metrics with the
   *same* `recall_at_fp_budget`/`grounding_stats` from the same saved arrays — no per-head metric
   drift.
5. **Conclusion is decisive either way:** the headline table answers the spec's success criterion —
   beat 0.300, or establish 0.300 as the feature ceiling. Both are reported in FINDINGS §9.

## Out of scope / deferred

- **Phase 3c attention pooling over Evo2 offsets** (torch) — conditional on 3a showing nonlinear
  heads help. Not built here.
- **Phase 5 LoRA / fine-tuning** — the "better features" direction, only relevant if §9 lands on
  the feature-ceiling conclusion.
- **Multi-seed confirmation of the §7 logistic headline** — a separate rigor task (the standing
  single-seed caveat), not part of head exploration.
