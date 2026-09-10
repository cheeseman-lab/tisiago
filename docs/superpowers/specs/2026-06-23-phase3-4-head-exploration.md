# Phase 3-4 -- Head Exploration: Beyond Logistic Regression

**Status:** Spec for Claude Code execution -- 2026-06-23
**Goal:** Improve on the 0.300 recall @ <=1 FP/tx ceiling by exploring richer classifiers
(Phase 3) and continuous efficiency regression (Phase 4). All experiments run on the existing
compact store -- no new GPU extraction needed.

---

## Context

The dense-trained logistic head (FINDINGS SS7-SS8) achieves:
- **recall 0.300 @ <=1 FP/tx** at true 230:1 imbalance (ag7 Dense(None), C=0.00075)
- **AUPRC 0.307** (71x the 0.0043 base rate)
- **non-cognate ~ 0.0003**

The autoresearch fleet previously found that MLPs all lost or crashed -- but that was on the
curated 3:1 set with 60k train samples. At dense scale (2M negatives, ~42k positives), richer
models may now have enough data to learn. The store infrastructure (compact gather + SSD) makes
iteration fast: each experiment is a CPU job, minutes to run.

**Training regime (settled, D1):** Always train on the dense TRAIN split at true imbalance
(~49:1 cognate), calibrate on dense VAL, evaluate on dense TEST at 230:1. Never curated 3:1.
This is the deployment distribution.

---

## Infrastructure

### What exists

```
data/dense_exp_store/              # compact store on NFS (4.3M rows, ag7 keys)
  manifest.parquet                 # row_idx, transcript_id, chrom, strand, codon,
                                   #   codon_class, split, label_tis
  embeddings/                      # memmap'd .npy per key
    alphagenome_jax/L16k/decoder_1bp/off0.npy     # [4329984, 1536] fp16
    alphagenome_jax/L131k/decoder_1bp/off0.npy    # [4329984, 1536] fp16
    evo2/W8k/blocks.28.mlp.l3/off{0,3,6,9}.npy   # [4329984, 4096] fp16 each
    onehot/kozakW20.npy                            # [4329984, 164] fp16
data/manifest.parquet              # curated manifest with efficiency columns
data/scan_manifest_allsplits.parquet  # 62.7M dense manifest (label_tis only)
src/tisiago/dense_caller.py        # current logistic head implementation
src/tisiago/caller.py              # metric functions: recall_at_fp_budget, reliability
src/tisiago/scan_eval.py           # grounding_stats
```

### Key functions to reuse

```python
from tisiago.caller import recall_at_fp_budget, reliability
from tisiago.scan_eval import grounding_stats
from tisiago.dense_caller import _load_full, _predict_chunked, evaluate, KEYS_7, KEYS_AG
```

`_load_full(emb_dir, keys, rows=None)` -> `[N, D]` float32 array. Preallocates and fills
column-block by column-block (no 2x peak from concatenate).

`evaluate(scan_store, heads, keys)` expects `heads` as a dict of
`{name: {"predict": callable, "scaler": ..., "model": ...}}`. The `predict` callable takes
`[N, D]` float32 and returns `[N]` probabilities. It handles chunked memmap scoring and
prints the full diagnostic table.

### Environment

`tisiago` conda env. **No xgboost/lightgbm/torch installed yet.** Install what you need:
```bash
conda activate tisiago
pip install xgboost lightgbm  # for tree-based classifiers
# torch only if needed for attention pooling (Phase 3c)
```

---

## Phase 3 -- Richer classifiers on the same features

All experiments below use the same 7-key feature stack (19,620-dim), the same train/val/test
splits, and the same evaluation protocol as SS7. The only change is the classifier.

### 3a. Gradient-boosted trees (XGBoost / LightGBM)

**Rationale:** XGBoost and LightGBM handle high-dimensional tabular data well, capture
feature interactions naturally, and are robust to feature scale (no StandardScaler needed).
They may find nonlinear interactions between the AG regional embedding and the Evo2
nucleotide embedding that a linear boundary misses.

**Implementation:** Create `src/tisiago/head_xgb.py`:

```python
"""Gradient-boosted tree heads for TIS classification."""

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

def fit_xgb_head(X_train, y_train, X_val, y_val,
                 n_estimators=500, max_depth=6, learning_rate=0.05,
                 scale_pos_weight=None, subsample=0.8,
                 colsample_bytree=0.5, seed=0):
    """Train an XGBoost classifier, isotonic-calibrate on val.

    Returns a dict compatible with dense_caller.evaluate():
      {"predict": callable, "model": ..., "scaler": None}
    """
    if scale_pos_weight is None:
        # auto: ratio of negatives to positives
        scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",        # fast for large N
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50)

    # Isotonic calibration on val
    cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)

    return {
        "predict": lambda X: cal.predict_proba(X)[:, 1],
        "model": clf,
        "calibrator": cal,
        "scaler": None,  # XGBoost doesn't need scaling
    }
```

**Hyperparameters to sweep** (small grid, not exhaustive):
- `max_depth`: [4, 6, 8]
- `n_estimators`: [300, 500, 1000] (with early stopping, this is a cap)
- `colsample_bytree`: [0.3, 0.5, 0.7] -- feature subsampling per tree
- `scale_pos_weight`: [auto (~49), 10, 1]

**Memory note:** XGBoost `hist` method builds histograms -- memory scales with
`n_features x n_bins x n_threads`, not with the full feature matrix. At 19.6k features,
expect ~20-40 GB peak. Run on a big-mem CPU node (`--mem=192G`).

**LightGBM variant** (same interface, may be faster):
```python
import lightgbm as lgb

clf = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.5,
    is_unbalance=True,
    random_state=seed,
    n_jobs=-1,
)
```

### 3b. Random forest baseline

**Rationale:** Sanity check -- if RF beats logistic but XGBoost doesn't, the gain is from
ensemble averaging, not interaction learning.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=0,
    n_jobs=-1,
)
# Calibrate
cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
```

**Memory note:** RF at 19.6k features x 2M rows x 500 trees is very large (~100+ GB).
Consider training on a 500k subsample first to see if it beats logistic at all.

### 3c. Attention pooling over Evo2 offsets (optional, if 3a succeeds)

**Rationale:** The 4 Evo2 offsets (off0, off3, off6, off9) are currently concatenated into
a 16,384-dim vector. But each offset sees a different downstream position -- the model could
learn which offset matters for each candidate. A small attention module pools across offsets
before concatenation with AG features.

```python
import torch
import torch.nn as nn

class OffsetAttentionHead(nn.Module):
    """Attention pool over K Evo2 offsets, then linear classify."""
    def __init__(self, evo_dim=4096, n_offsets=4, ag_dim=3072+164, hidden=128):
        super().__init__()
        # Attention over offsets
        self.query = nn.Linear(evo_dim, hidden)
        self.key = nn.Linear(evo_dim, hidden)
        # Classifier
        self.head = nn.Linear(evo_dim + ag_dim, 1)

    def forward(self, evo_offsets, ag_features):
        # evo_offsets: [B, n_offsets, evo_dim]
        Q = self.query(evo_offsets)       # [B, K, H]
        K = self.key(evo_offsets)         # [B, K, H]
        attn = (Q * K).sum(-1) / (128 ** 0.5)  # [B, K]
        attn = torch.softmax(attn, dim=-1)      # [B, K]
        pooled = (attn.unsqueeze(-1) * evo_offsets).sum(1)  # [B, evo_dim]
        x = torch.cat([pooled, ag_features], dim=-1)
        return self.head(x).squeeze(-1)
```

**Only pursue this if 3a shows that nonlinear heads help.** If XGBoost doesn't beat logistic,
attention pooling won't either -- the features are the ceiling.

---

## Phase 4 -- Continuous efficiency regression

### 4a. Label construction

The curated manifest has per-cell-line efficiency columns:
```
max_norm_HeLa     (float64, 174k nulls, range [0, 3457])
max_norm_K562     (float64, 167k nulls, range [0, 1360])
max_norm_RPE1_*   (float64, 175-180k nulls)
max_norm_U2OS     (float64, 175k nulls)
```

These are only on the **curated** manifest (192k candidates) -- the dense scan manifest has
`label_tis` (binary) only. So regression training uses the curated store, not the dense store.

**Target construction options:**

1. **Per-line regression** -- predict `max_norm_HeLa` directly. Simple, interpretable, but
   HeLa-specific. ~17.8k non-null values (positives with measured efficiency).

2. **Mean efficiency across lines** -- `mean(max_norm_HeLa, max_norm_K562, ...)` where
   non-null. More robust, less cell-type-specific.

3. **Log-transformed** -- the efficiency values span 0-3457 with a heavy right tail. Use
   `log1p(max_norm)` as the target. Standard practice for count-like data.

4. **Binary-plus-regression** -- two-stage: first classify TIS/not-TIS (the current task),
   then regress efficiency conditional on being a TIS. Separates the two sub-problems.

**Recommendation:** Start with option 3 (`log1p(max_norm_HeLa)`) on the curated store.
The curated store already has all 7 keys for all 192k candidates. This is a fast experiment.

### 4b. Regression head implementation

Create `src/tisiago/efficiency_head.py`:

```python
"""Efficiency regression head -- predict continuous initiation efficiency."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

def fit_efficiency_head(X_train, y_train, X_val, y_val, alpha=1.0):
    """Train a Ridge regressor on log1p(max_norm_HeLa).

    Returns a dict with predict callable + metrics.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)

    model = Ridge(alpha=alpha)
    model.fit(Xtr, y_train)

    pred_val = model.predict(Xva)
    metrics = {
        "r2_val": r2_score(y_val, pred_val),
        "rmse_val": mean_squared_error(y_val, pred_val) ** 0.5,
        "spearman_val": spearmanr(y_val, pred_val).correlation,
    }

    def predict(X):
        return model.predict(scaler.transform(X))

    return {"predict": predict, "model": model, "scaler": scaler, "metrics": metrics}


def evaluate_as_classifier(efficiency_preds, y_binary, tx, budgets=(1.0, 5.0, 20.0)):
    """Threshold continuous efficiency predictions to recover classification metrics.

    Higher predicted efficiency -> more likely to be a TIS. This lets us directly compare
    the regression head to the classification heads on recall@FP/tx.
    """
    from tisiago.caller import recall_at_fp_budget
    from sklearn.metrics import average_precision_score

    # Normalize predictions to [0,1] range for threshold-based metrics
    p_min, p_max = efficiency_preds.min(), efficiency_preds.max()
    p_norm = (efficiency_preds - p_min) / (p_max - p_min + 1e-10)

    results = {"AUPRC": average_precision_score(y_binary, p_norm)}
    for b in budgets:
        r = recall_at_fp_budget(p_norm, y_binary, tx, budget=b)
        results[f"recall@{b}FP"] = r["recall"]
    return results
```

### 4c. Training protocol

```python
# Load curated store
emb = Path("data/store/embeddings")
m = pd.read_parquet("data/store/manifest.parquet")

# Target: log1p(max_norm_HeLa), only where measured
has_eff = m["max_norm_HeLa"].notna()
y_eff = np.log1p(m.loc[has_eff, "max_norm_HeLa"].values)

# For non-TIS candidates with null efficiency: assign 0 (no initiation)
# This lets us train on all candidates, not just positives
y_full = np.zeros(len(m))
y_full[has_eff.values] = y_eff

# Split
tr = m.split.values == "train"
va = m.split.values == "val"
te = m.split.values == "test"

X = _load_full(emb, KEYS_7)
head = fit_efficiency_head(X[tr], y_full[tr], X[va], y_full[va], alpha=1.0)

# Evaluate as classifier on the dense test set
# Use the regression head's predict() as a score -> threshold for recall@FP
```

**The key question:** Does predicting *how much* initiation (continuous) produce a better
*ranking* of candidates than predicting *whether* initiation (binary)? If the regression
head's AUPRC on the dense test set exceeds 0.307, it's finding signal the binary head misses.

### 4d. Ridge alpha sweep

```python
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    head = fit_efficiency_head(X[tr], y_full[tr], X[va], y_full[va], alpha=alpha)
    print(f"alpha={alpha:>7.1f}  R2={head['metrics']['r2_val']:.4f}  "
          f"rho={head['metrics']['spearman_val']:.4f}")
```

---

## Execution plan

### Step 1: Install dependencies
```bash
conda activate tisiago
pip install xgboost lightgbm
```

### Step 2: Run logistic baseline at 2M (reproduce SS7 headline as reference)
```bash
cd /path/to/tisiago
python -m tisiago.dense_caller --train dense --features ag7 --save-preds data/preds_logistic.npz
```
This is the 0.300 recall baseline. Save predictions so metrics are instant for comparison.

### Step 3: XGBoost sweep (Phase 3a)
Write `src/tisiago/head_xgb.py` as above. Integrate into `dense_caller.py`:
- Add `--head` argument: `{logistic, xgboost, lightgbm, ridge_regression}`
- When `--head xgboost`: call `fit_xgb_head` instead of `fit_calibrated_head`
- Same `evaluate()` call -- the interface is `{"predict": callable}`

Run on a big-mem CPU node:
```bash
srun --partition=20 --mem=192G --cpus-per-task=16 --time=4:00:00 \
    python -m tisiago.dense_caller --train dense --features ag7 --head xgboost \
    --save-preds data/preds_xgb.npz
```

If XGBoost beats logistic, sweep hyperparameters:
```bash
for depth in 4 6 8; do
  for lr in 0.01 0.05 0.1; do
    python -m tisiago.dense_caller --train dense --features ag7 --head xgboost \
      --xgb-depth $depth --xgb-lr $lr --save-preds data/preds_xgb_d${depth}_lr${lr}.npz
  done
done
```

### Step 4: Efficiency regression (Phase 4a-c)
Write `src/tisiago/efficiency_head.py` as above. Run:
```bash
python -m tisiago.efficiency_head --store data/store --features ag7 \
    --target max_norm_HeLa --save-preds data/preds_ridge.npz
```

Then evaluate the regression predictions as a classifier on the dense test set -- load the
saved preds, apply `evaluate_as_classifier` against the dense test labels.

### Step 5: Compare all heads
Build a comparison table:

| Head | AUPRC | recall@1FP | recall@5FP | recall@20FP | non-cog mean p |
|---|---|---|---|---|---|
| Logistic (SS7 baseline) | 0.307 | 0.300 | 0.539 | 0.769 | 0.0003 |
| XGBoost (best) | ? | ? | ? | ? | ? |
| LightGBM (best) | ? | ? | ? | ? | ? |
| Ridge efficiency | ? | ? | ? | ? | ? |

### Step 6: Update FINDINGS.md
Add SS9 with the comparison table and interpretation.

---

## Success criteria

- **Beat 0.300 recall @ <=1 FP/tx** with any head on the same ag7 features at 230:1.
- **Or establish that 0.300 is the feature ceiling** -- if XGBoost, LightGBM, and regression
  all plateau at ~0.30, the linear head isn't the bottleneck, and the next step is better
  features (Phase 5 LoRA) not better classifiers.
- Either result is informative and worth reporting.

## Key constraints

- **Train on dense, always.** The curated 3:1 set is only for the regression target (Phase 4)
  because efficiency labels only exist there. Classification training is always dense.
- **Same evaluation protocol.** All heads evaluated on the same dense TEST @ 230:1 with the
  same `recall_at_fp_budget` and `grounding_stats` functions.
- **No GPU needed.** Everything runs on the compact store (SSD or NFS). CPU-only.
- **Save predictions.** Every head saves its test predictions as `.npz` so any future metric
  is instant without re-scoring.

## Files to create/modify

- `src/tisiago/head_xgb.py` -- XGBoost/LightGBM head (NEW)
- `src/tisiago/efficiency_head.py` -- Ridge regression head (NEW)
- `src/tisiago/dense_caller.py` -- add `--head` argument, integrate new heads
- `FINDINGS.md` -- add SS9 with comparison table
