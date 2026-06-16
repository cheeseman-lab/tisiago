# Phase 1: Calibrated Caller Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `src/tisiago/caller.py` that turns the existing logistic ranking head into a *calibrated* probability and scores it with caller-shaped metrics (reliability, Brier, recall @ false-positives-per-transcript budget), on the store that already exists.

**Architecture:** Pure-CPU evaluation over `data/store`. Train the logistic head on the `train` split (as `eval.py` does), fit an isotonic calibrator on the held-out `val` split (chr7, currently unused by any head), evaluate on `test` (chr8/chr9). Metric functions take plain arrays (`p`, `y`, `transcript_id`) so Phase 2's dense scan reuses them unchanged. Existing `eval.py`/`resolution.py` stay as the PoC ranking sanity checks.

**Tech Stack:** numpy, pandas, scikit-learn (LogisticRegression, IsotonicRegression, brier_score_loss), pytest. Runs in the `tisiago` conda env (sklearn 1.9, numpy 2.4).

**Environment for every command below:**
```bash
eval "$(conda shell.bash hook)" && conda activate tisiago
```

---

## File Structure

- **Create `src/tisiago/caller.py`** — calibration + caller metrics + a `main()` CLI. Three pure metric functions (`recall_at_fp_budget`, `reliability`, a tiny train/calibrate helper) plus a `__main__` that prints the report. One responsibility: deliverable-track (calibrated, caller-shaped) evaluation.
- **Create `tests/test_caller.py`** — unit tests for the pure metric functions on tiny synthetic arrays (no store, no GPU, fast).
- **Modify `ARCHITECTURE.md`** — add `caller.py` to the module map and a short "evaluation tracks" note (PoC ranking vs deliverable caller).
- **Unchanged:** `eval.py`, `resolution.py`, `tiling.py`, `extract.py`, `store.py`.

The metric functions live in `caller.py` (not a new `metrics.py`) because they are small and only used here in Phase 1; Phase 2 will import them from `caller.py`. Split out a `metrics.py` only if Phase 2 grows a second consumer.

---

## Task 1: Pure metric — `recall_at_fp_budget`

The headline metric. Given calibrated probabilities `p`, binary labels `y`, and a
per-row `transcript_id`, sweep a global threshold and find the operating point where the
mean number of false positives per transcript is at most `budget`; return recall and the
threshold there.

**Files:**
- Create: `src/tisiago/caller.py`
- Test: `tests/test_caller.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_caller.py
import numpy as np
from tisiago.caller import recall_at_fp_budget


def test_recall_at_fp_budget_perfect_separation():
    # 2 transcripts, perfectly separable: all positives score above all negatives.
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.85, 0.05, 0.15])
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    tx = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    res = recall_at_fp_budget(p, y, tx, budget=1.0)
    # A threshold exists that admits all 4 positives and 0 false positives.
    assert res["recall"] == 1.0
    assert res["fp_per_transcript"] <= 1.0
    assert 0.2 < res["threshold"] <= 0.8


def test_recall_at_fp_budget_respects_budget():
    # One transcript, 1 positive and 4 negatives just below it.
    # Budget 0 forbids any FP -> threshold must sit above the top negative,
    # so recall can still be 1.0 only if the positive outscores every negative.
    p = np.array([0.6, 0.5, 0.4, 0.3, 0.2])
    y = np.array([1, 0, 0, 0, 0])
    tx = np.array(["A", "A", "A", "A", "A"])
    res = recall_at_fp_budget(p, y, tx, budget=0.0)
    assert res["recall"] == 1.0
    assert res["fp_per_transcript"] == 0.0
    assert res["threshold"] > 0.5  # above the highest negative


def test_recall_at_fp_budget_returns_keys():
    p = np.array([0.7, 0.3])
    y = np.array([1, 0])
    tx = np.array(["A", "A"])
    res = recall_at_fp_budget(p, y, tx, budget=1.0)
    assert set(res) == {"recall", "threshold", "fp_per_transcript", "budget"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_caller.py -v`
Expected: FAIL — `ImportError: cannot import name 'recall_at_fp_budget'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tisiago/caller.py
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

import numpy as np


def recall_at_fp_budget(p, y, transcript_id, budget: float = 1.0) -> dict:
    """Recall at the threshold whose mean false-positives-per-transcript ≤ budget.

    A false positive is a negative codon scored ≥ threshold. The budget is the mean,
    over distinct transcripts, of such false positives. We pick the *lowest* threshold
    (highest recall) that still satisfies the budget.

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

    # Candidate thresholds: just above each negative's score admits everything strictly
    # higher. Sort unique negative scores descending; sweep from strict (high) to loose.
    neg_scores = np.unique(p[~y])
    # thresholds to try: each negative score (loose enough to admit scores > it),
    # plus +inf sentinel (admit nothing). Use scores themselves as ">=" cutoffs.
    candidates = np.concatenate([neg_scores, [np.inf]])
    candidates.sort()  # ascending: high threshold (few FP) -> low threshold (many FP)
    candidates = candidates[::-1]

    best = {"recall": 0.0, "threshold": float("inf"), "fp_per_transcript": 0.0,
            "budget": float(budget)}
    for thr in candidates:
        admitted = p >= thr
        fp = admitted & ~y
        # mean FP per transcript across ALL transcripts present (not just those with FP)
        fp_per_tx = fp.sum() / max(1, n_tx)
        if fp_per_tx <= budget:
            recall = (admitted & y).sum() / max(1, n_pos)
            if recall >= best["recall"]:
                best = {"recall": float(recall), "threshold": float(thr),
                        "fp_per_transcript": float(fp_per_tx), "budget": float(budget)}
        else:
            break  # thresholds only get looser -> FP only grows
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_caller.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tisiago/caller.py tests/test_caller.py
git commit -m "feat(caller): recall at false-positives-per-transcript budget"
```

---

## Task 2: Pure metric — `reliability`

Binned calibration curve + Brier score, to show calibration improves `p`.

**Files:**
- Modify: `src/tisiago/caller.py`
- Test: `tests/test_caller.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_caller.py
from tisiago.caller import reliability


def test_reliability_perfectly_calibrated():
    # p exactly equals empirical frequency in each bin -> low Brier, small gap.
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=20000)
    y = (rng.uniform(0, 1, size=20000) < p).astype(int)  # P(y=1) = p by construction
    res = reliability(p, y, n_bins=10)
    assert res["brier"] < 0.20
    assert res["max_gap"] < 0.05          # |confidence - accuracy| per bin
    assert len(res["bin_confidence"]) == len(res["bin_accuracy"]) == 10


def test_reliability_overconfident_has_large_gap():
    # Always predict 0.99 but only half are positive -> big calibration gap.
    p = np.full(1000, 0.99)
    y = np.array([1, 0] * 500)
    res = reliability(p, y, n_bins=10)
    assert res["brier"] > 0.4
    assert res["max_gap"] > 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_caller.py::test_reliability_perfectly_calibrated -v`
Expected: FAIL — `ImportError: cannot import name 'reliability'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/tisiago/caller.py
from sklearn.metrics import brier_score_loss


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_caller.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tisiago/caller.py tests/test_caller.py
git commit -m "feat(caller): reliability curve + Brier score"
```

---

## Task 3: Train + calibrate helper

Wrap the train→calibrate→predict flow so `main()` and tests share one path. Trains a
standardized logistic head, then fits isotonic regression on a held-out calibration set.

**Files:**
- Modify: `src/tisiago/caller.py`
- Test: `tests/test_caller.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_caller.py
from tisiago.caller import fit_calibrated_head


def test_fit_calibrated_head_improves_brier_on_separable_data():
    # Two well-separated Gaussian blobs in 5-D. Raw logistic is already decent;
    # isotonic calibration should not worsen Brier on a held-out calibration split.
    rng = np.random.default_rng(0)
    d = 5
    def blob(center, n):
        return rng.normal(center, 1.0, size=(n, d))
    Xtr = np.vstack([blob(0, 400), blob(3, 400)])
    ytr = np.array([0] * 400 + [1] * 400)
    Xcal = np.vstack([blob(0, 200), blob(3, 200)])
    ycal = np.array([0] * 200 + [1] * 200)
    Xte = np.vstack([blob(0, 200), blob(3, 200)])
    yte = np.array([0] * 200 + [1] * 200)

    head = fit_calibrated_head(Xtr, ytr, Xcal, ycal)
    p_raw = head["predict_raw"](Xte)
    p_cal = head["predict"](Xte)
    assert p_raw.shape == (400,)
    assert p_cal.shape == (400,)
    assert ((p_cal >= 0) & (p_cal <= 1)).all()
    # On separable data both are good; calibrated Brier should be close or better.
    from sklearn.metrics import brier_score_loss
    assert brier_score_loss(yte, p_cal) <= brier_score_loss(yte, p_raw) + 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_caller.py::test_fit_calibrated_head_improves_brier_on_separable_data -v`
Expected: FAIL — `ImportError: cannot import name 'fit_calibrated_head'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/tisiago/caller.py
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit_calibrated_head(X_train, y_train, X_cal, y_cal) -> dict:
    """Train a standardized logistic head, then isotonic-calibrate on a held-out set.

    Args:
        X_train, y_train: training features/labels for the logistic head.
        X_cal, y_cal: held-out calibration features/labels (a different chromosome split).

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_caller.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tisiago/caller.py tests/test_caller.py
git commit -m "feat(caller): standardized logistic head + isotonic calibration"
```

---

## Task 4: `main()` CLI — the report on the real store

Tie it together: load the store, train on `train`, calibrate on `val`, evaluate on
`test`, print reliability (raw vs calibrated) + recall @ FP/transcript budget. Mirrors
`eval.py`'s arg/loading style (`--store`, `--keys`, the `load()` helper).

**Files:**
- Modify: `src/tisiago/caller.py`

- [ ] **Step 1: Add the loader + `main()`**

```python
# add to src/tisiago/caller.py
import argparse
from pathlib import Path

import pandas as pd

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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="data/store", help="Path to the assembled store.")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS,
                    help="Feature .npy keys to concatenate (default: AG16k + Evo2 blk28).")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="Max mean false positives per transcript for the headline metric.")
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
    print(f"train={len(tr)} (of {len(tr_all)})  cal/val={len(cal)}  test={len(te)}  "
          f"test transcripts={len(np.unique(tx_te))}  test pos-rate={yte.mean():.3f}")
    print("\n-- calibration (held-out test) --")
    for name, p in [("raw logistic", p_raw), ("isotonic-calibrated", p_cal)]:
        r = reliability(p, yte, n_bins=10)
        print(f"  {name:22s} Brier={r['brier']:.4f}  max_gap={r['max_gap']:.3f}")

    print(f"\n-- caller metric (calibrated, budget ≤ {args.budget} FP/transcript) --")
    res = recall_at_fp_budget(p_cal, yte, tx_te, budget=args.budget)
    print(f"  recall={res['recall']:.3f}  at threshold p≥{res['threshold']:.3f}  "
          f"(actual {res['fp_per_transcript']:.3f} FP/transcript)")
    print("\n  NOTE: on the curated 3:1 decoy set, NOT true genome-wide imbalance.")
    print("  The true-imbalance number requires the Phase 2 dense codon scan.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the report on the real store**

Run: `python -m tisiago.caller --store data/store`
Expected: prints feature/split header; a calibration block where **isotonic-calibrated
Brier ≤ raw logistic Brier** on test; and a recall @ FP/transcript line with a threshold
in (0, 1). No exceptions.

- [ ] **Step 3: Verify the metric functions still pass (no regression)**

Run: `pytest tests/test_caller.py -v`
Expected: PASS (6 tests).

- [ ] **Step 4: Lint**

Run: `ruff check src/tisiago/caller.py && ruff format --check src/tisiago/caller.py`
Expected: no errors (run `ruff format src/tisiago/caller.py` first if needed).

- [ ] **Step 5: Commit**

```bash
git add src/tisiago/caller.py
git commit -m "feat(caller): CLI report — calibrate on val, caller metrics on test"
```

---

## Task 5: Update ARCHITECTURE.md

Reflect the new module and the two evaluation tracks.

**Files:**
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1: Add `caller.py` to the §2 module map table**

In `ARCHITECTURE.md`, in the "Module map" code block, add after the `resolution.py` line:
```
└── caller.py      head (CPU)         .npy ──▶ calibrated p ──▶ reliability · recall @ FP/transcript budget
```
And add a row to the module table directly below the `resolution.py` row:
```
| `caller.py` | Calibrated probability + caller-shaped metrics (deliverable track) | numpy/pandas/sklearn | runs on store, no GPU |
```

- [ ] **Step 2: Add an "evaluation tracks" note under the table**

Add this sentence after the "The cut that matters" paragraph in §2:
```
**Two evaluation tracks:** `eval.py` / `resolution.py` measure *ranking* (AUROC,
win-rate) — the PoC sanity check; `caller.py` measures the *deliverable* — a calibrated
probability and recall at a false-positives-per-transcript budget. Phase 2's dense scan
reuses `caller.py`'s metric functions unchanged.
```

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs(architecture): add caller.py + evaluation-tracks note"
```

---

## Self-Review Notes

- **Spec coverage:** Phase 1 §Components → Tasks 3 (train+calibrate), 1 (recall@FP budget), 2 (reliability), 4 (CLI report wiring train/val/test). Honest-scope boundary → printed NOTE in Task 4 + plan goal. Architecture impact (`caller.py` added, `eval.py`/`resolution.py` kept) → Task 5. Phase 2 / Phase 3 are intentionally out of this plan.
- **Type consistency:** `fit_calibrated_head` returns `{"predict", "predict_raw"}` (callables) — used identically in Task 3 test and Task 4 `main`. `recall_at_fp_budget` returns keys `{recall, threshold, fp_per_transcript, budget}` — asserted in Task 1, consumed in Task 4. `reliability` returns `{brier, max_gap, bin_confidence, bin_accuracy}` — asserted in Task 2, consumed in Task 4.
- **Placeholders:** none — every code step is complete and runnable.
- **Out of scope (correctly):** true-imbalance numbers, non-cognate grounding, dense scan — all Phase 2; cross-validation — Phase 3.
