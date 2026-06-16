# Phase 2B — Train at true imbalance (Option B)

**Date:** 2026-06-16
**Status:** design, pre-implementation
**Depends on:** Phase 2 scan store (assembled `data/scan_store/` with AG16k + Evo2 keys); Phase 1 `caller.py` (metric functions reused).
**Supersedes:** The curated-trained-then-applied evaluation in `scan_eval.py`. That module stays as a **baseline** — this phase adds a new training regime alongside it.

## Motivation

Phase 2 proved the gap: a head trained on the curated 3:1 decoy set achieves recall 0.094 @ ≤1 FP/transcript when deployed at the true 230:1 imbalance (AG-only). The ranking signal is real (AUROC 0.90), but the decision boundary is in the wrong place because the head learned it from a distribution 77× narrower than reality. Non-cognate codons — never trained on — receive mean p = 0.12 instead of ≈0.

The fix is to train on the distribution the head will be evaluated on: the dense scan manifest, at true imbalance, with class weighting to keep the rare positives from drowning. This is the same principle that makes AlphaGenome's supervised heads work — they train on the real genome-wide distribution, not a curated slice.

## What changes

One new module: `src/tisiago/dense_caller.py`. Everything else — the scan store, the embedding arrays, the metric functions — is reused from Phase 1 and Phase 2.

### Training data

**Source:** the assembled scan store, which is row-aligned to `data/scan_manifest.parquet`.

**Splits (same chromosome scheme):**
- **Train:** all scan-manifest rows where `split == "train"`. These come from `enumerate_codons.py` applied to the train-chromosome transcripts. This requires a **train-split scan manifest + scan store**, which Phase 2 only produced for test+val. The first implementation step is to enumerate + embed train transcripts (same pipeline, just pointed at the train split). Alternatively, for a fast first experiment, use the *curated* train manifest at 3:1 for training and a *dense val* for calibration (see Phased approach below).
- **Val (calibration):** `split == "val"` (chr7) rows from the scan manifest — dense, all codons. The isotonic calibrator sees the true distribution.
- **Test:** `split == "test"` (chr8/chr9) — the same 5M-position dense scan already assembled.

**Row composition at training time (true imbalance):**
- Every AUG, near-cognate, and non-cognate codon in the train transcripts.
- Labels: `label_tis == 1` for Ribo-seq–called TIS, 0 for everything else.
- Expected imbalance: ~200–300:1 (negatives per positive), matching test.

### The head

```python
LogisticRegression(
    max_iter=1000,        # bumped from 300 — convergence warning at 230:1
    C=1.0,
    class_weight="balanced",   # THE key change: sklearn auto-weights inversely
                               # proportional to class frequency
    solver="lbfgs",
)
```

`class_weight="balanced"` makes sklearn set `w_pos = N / (2 * n_pos)` and `w_neg = N / (2 * n_neg)`. At 230:1 this upweights each positive by ~115×, so the loss gradient from positives isn't drowned by the vast negative pool. The boundary shifts to accommodate the real class ratio.

**Why not focal loss / MLP / custom architecture yet:** Phase 3 (autoresearch) is the right place for architecture sweeps. This phase should change exactly one variable — the training distribution — so we can measure its effect cleanly. A logistic head with balanced class weights is the minimal change from the current pipeline.

### Calibration

Isotonic calibration on the **dense val split** (all codons in chr7 transcripts from the scan manifest). This is the other critical change: the calibrator now sees the true score distribution, not the curated 3:1 distribution. Because isotonic is monotonic, the ranking is preserved — but the probability mapping will be very different.

### Evaluation (unchanged metrics, new training)

Metric functions from `caller.py` and `scan_eval.py` are reused as-is:
- `caller.recall_at_fp_budget(p, y, transcript_id, budget=1.0)` on cognate test codons.
- `caller.reliability(p, y)` for Brier / calibration curve.
- `scan_eval.grounding_stats(p_noncognate, threshold)` on non-cognate test codons.

The only change is what produced `p` — the dense-trained head instead of the curated-trained head.

### Training subsample

The curated regime subsamples 60k train rows (`TRAIN_SUBSAMPLE`). At true imbalance, the train set will be millions of rows (every codon in train transcripts). Options:

1. **Full train set** — logistic regression on standardized features is O(N·D·iters). At N=~30M, D=5632, this is substantial but feasible on CPU (~10–30 min with l-bfgs). Try this first.
2. **Subsample negatives, keep all positives** — if (1) is too slow, sample ~500k negatives + all ~30k positives for a ~17:1 ratio. Still far more representative than 3:1. Record the subsample ratio.
3. **Stratified subsample** — sample negatives proportionally by codon_class (keep the AUG/near-cognate/non-cognate ratio). This preserves the distribution shape even at reduced N.

Start with (1). Fall back to (2) or (3) only on convergence or memory issues — and record which was used.

## Module: `src/tisiago/dense_caller.py`

```
dense_caller.py
├── fit_dense_head(scan_store, curated_store, keys, subsample=None)
│     Load train-split embeddings from scan_store (or curated + dense val — see Phased approach).
│     StandardScaler → LogisticRegression(class_weight="balanced", max_iter=1000).
│     Isotonic calibrate on dense val.
│     Return {predict, predict_raw, head_info} (head_info: N_train, pos_rate, class_weights, converged).
│
├── evaluate_dense(head, scan_store, keys, budget=1.0)
│     Score test-split scan rows. Compute:
│       - recall_at_fp_budget (cognate test, true imbalance)
│       - grounding_stats (non-cognate test)
│       - reliability (cognate test, for Brier/calibration curve)
│       - AUROC/AUPRC on cognate test (for back-reference to Phase 1 ranking metrics)
│     Print results. Return results dict.
│
└── main()
      CLI: --scan-store, --curated-store, --keys, --budget
      Calls fit_dense_head → evaluate_dense.
      Also runs the OLD curated-trained head (from caller.fit_calibrated_head)
      on the same test set, for a side-by-side comparison table.
```

### Output format

```
===== DENSE-TRAINED GLOBAL CALLER (Option B) =====
training: N=31,209,344  pos=29,112  neg=31,180,232  ratio=1071:1
  class_weight=balanced  max_iter=1000  converged=True
calibration: dense val (chr7), N=...

test cognate codons: 823,302  non-cognate: 4,196,378
true imbalance (neg:pos over cognate test): 230.6:1

                             curated-trained    dense-trained
  recall @ ≤1 FP/tx             0.094              ???
  threshold p>=                  0.885              ???
  AUROC (cognate test)           ???                ???
  AUPRC (cognate test)           ???                ???

NON-COGNATE GROUNDING:
                             curated-trained    dense-trained
  mean p                        0.119              ???
  p95                           0.483              ???
  FPR @ threshold               0.0016             ???
```

The side-by-side is the deliverable — it answers "did training at true imbalance help?" in one table.

## Phased approach (fast experiment first, full pipeline second)

### 2B.1 — Quick experiment (no new GPU work)

The scan store already has test + val embeddings. The curated store has train embeddings. A fast first experiment:

1. Train the balanced-logistic head on the **curated train** (3:1, same as today) — no change here.
2. Isotonic-calibrate on the **dense val** (all codons in chr7 from the scan store) instead of the curated val.
3. Evaluate on the dense test (same as current scan_eval).

This tests whether the recall collapse is primarily a **calibration** problem (the isotonic calibrator was fitted on 3:1) vs. a **boundary** problem (the logistic weights themselves are wrong). If dense-val calibration alone recovers significant recall, that's informative — and it needs zero new GPU extraction.

### 2B.2 — Full dense training

Requires train-split scan manifest + scan store:
1. `python -m tisiago.enumerate_codons` on train transcripts → `data/scan_manifest_train.parquet`.
2. GPU extraction (same SLURM arrays) → `data/scan_parts_train/`.
3. Assembly → `data/scan_store_train/`.
4. Train `dense_caller.py` on the full dense train set.

The GPU cost is comparable to the test+val scan already done (~similar transcript count on train chromosomes). This is the principled, full Option B.

**Recommendation:** Run 2B.1 first (today, CPU-only, ~30 min). If the improvement is modest, proceed to 2B.2 (requires a GPU extraction round, ~3–4 hr). Report both.

## Metrics / done when

1. `dense_caller.py` prints the side-by-side table (curated-trained vs. dense-trained) on the same test set.
2. Recall @ ≤1 FP/transcript improves over the curated-trained baseline (0.094 AG-only).
3. Non-cognate grounding improves (mean p closer to 0).
4. If 2B.1 (dense-val calibration only) already recovers recall, report that finding — it means the boundary was adequate but the calibration mapping was wrong.
5. Results recorded in `FINDINGS.md` §5.

## Risks

- **Train-set scan store size.** Train chromosomes have more transcripts than test+val. At ~60 GB for test-only, the full train scan store could be ~100–150 GB. Manageable on `/lab/barcheese01/` but note the disk footprint. Can restrict to headline keys only (AG16k + Evo2 blk28).
- **Logistic convergence at scale.** l-bfgs on ~30M × 5632 features may be slow. `max_iter=1000` should suffice; if not, try `saga` solver with `tol=1e-3` or subsample negatives (option 2 above).
- **Overfitting to non-cognate trivial negatives.** At true imbalance, ~80% of negatives are non-cognate (trivially non-initiating). The head might learn "reject non-ATG/near-cognate" as a shortcut — which is correct biology but wouldn't generalize if we later want to score only cognate codons. **Mitigation:** the eval already stratifies cognate vs. non-cognate, so this would be visible. And `class_weight="balanced"` weights by label, not by codon class, so the logistic objective is still about initiation vs. non-initiation.
- **Phase 3 interaction.** This phase deliberately keeps the architecture fixed (logistic, balanced weights) so the effect of the training distribution is isolated. Phase 3's architecture sweep should start from the dense-trained head, not the curated one.

## Files touched

| File | Change |
|---|---|
| `src/tisiago/dense_caller.py` | **new** — fit_dense_head, evaluate_dense, main |
| `src/tisiago/caller.py` | bump `max_iter` 300→1000 (convergence fix, backwards-compatible) |
| `scripts/run_tis_scan.sh` | add a `--split train` mode if needed for 2B.2 |
| `FINDINGS.md` | new §5 with the side-by-side table |
| `ROADMAP.md` | update P2 status |
| `tests/test_dense_caller.py` | **new** — unit tests for fit_dense_head, evaluate_dense |

## Out of scope

- MLP / architecture changes (Phase 3).
- Efficiency regression (Phase 4).
- Genome-wide deployment beyond test+val+train transcripts.
- Changing the embedding keys or adding new layers — this phase changes only the training regime.

