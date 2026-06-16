# Integrating the general codon→TIS predictor

**Date:** 2026-06-16
**Status:** design, pre-implementation
**Direction encoded in:** `CLAUDE.md` → *Direction — a general codon→TIS predictor*

## Goal

Move tisiago from a **candidate-ranker** (real TIS vs 3:1 curated decoys) toward a
**general codon → P(initiation) predictor**: any codon in an expressed transcript →
a *calibrated* probability of being a translation-initiation site, including a
confident "no" for non-starts. Build from the ground up, broad first, then narrow.

This spec covers the integration arc. **The first implementation plan targets Phase 1
only** — it is pure CPU on the store that already exists and needs no new data. Phase 2
is scoped here but has one unresolved data dependency (see §Phase 2) and gets its own
spec once Phase 1 lands.

## Why now / what's missing

The PoC proved the signal exists and is linearly decodable (0.90 AUROC, 0.82
near-neighbour win-rate; `FINDINGS.md`). Three things block "general":

1. **No confident "no".** Ranking metrics (AUROC, win-rate) never require the model to
   *reject* anything — a decoy at p=0.45 losing to a positive at p=0.6 scores well while
   rejecting nothing. We have no calibrated probability and no specificity number.
2. **Curated, not dense.** It scores swissisoform's curated candidates, not every codon.
3. **No grounding control.** Nothing checks that biologically-impossible starts
   (non-cognate codons) are driven to ≈0.

---

## Phase 1 — Calibrate + caller-shaped metrics (CPU, existing store)

**Outcome:** the existing logistic head, but emitting a *calibrated* probability, scored
with caller-shaped metrics instead of pure ranking. First real "rejects plausible
non-starts" number, on the near-cognate decoys already in the store.

### Components

New module **`src/tisiago/caller.py`** (the PoC `eval.py` / `resolution.py` stay as-is —
they are the ranking sanity check; `caller.py` is the deliverable-track evaluation).

1. **Train** the logistic head on the `train` split (unchanged from `eval.py`).
2. **Calibrate** on the `val` split (chr7, 8,438 rows — *currently unused by any head*).
   Fit a probability calibrator (isotonic, with sigmoid/Platt as a fallback for the
   smaller strata) mapping raw scores → calibrated p. Calibrating on held-out val, not
   train, is what makes p mean something on test.
3. **Evaluate** on the `test` split (chr8/chr9) and report:
   - **Reliability** (binned calibration curve) + **Brier score**, before vs after
     calibration — does p mean what it says?
   - **Recall @ FP-per-transcript budget** (headline). Sweep a global threshold τ; a
     false positive = a negative codon with p ≥ τ; FP/transcript = mean over test
     transcripts of negatives clearing τ. Report recall (fraction of positives clearing
     τ) at the τ where mean FP/transcript ≤ B (default B=1), plus τ and the calibrated p
     at that point.
   - **AUPRC** at the curated 3:1 rate, for continuity with `FINDINGS.md`.

### Honest scope boundary (important)

Phase 1 implements the **metric machinery** and gives a first number, but on the curated
**3:1** test set. Two things it *cannot* yet deliver, by construction:

- **True genome-wide imbalance.** The store only holds 3 decoys per positive; the real
  near-cognate:positive ratio per transcript is much larger. Phase 1 reports the metric on
  the curated set and *may* add an importance-weighted projection to higher imbalance, but
  the true-imbalance number requires the full per-transcript codon enumeration → Phase 2.
- **Non-cognate grounding.** No non-cognate codons are in the store → the ≈0 control is
  Phase 2 only.

`caller.py` is written so the same metric functions consume Phase 2's dense enumeration
unchanged — Phase 2 swaps the *input rows*, not the metric code.

### Done when

- `python -m tisiago.caller --store data/store` prints: reliability + Brier
  (pre/post-calibration), and recall @ FP/transcript ≤ 1 with its τ, on held-out test.
- Calibration measurably improves Brier on test.
- The metric functions are import-clean for reuse (no test-set assumptions baked in).

---

## Phase 2 — Dense codon scan + non-cognate grounding (GPU, new extraction)

**Outcome:** score *all 64 codons* across expressed transcripts; evaluate as a caller at
true imbalance on near-cognate decoys, and as a grounding control on held-out non-cognate
codons (expect mean p ≈ 0).

### Components (sketch — own spec later)

- **Enumeration** (`src/tisiago/scan.py`): given an expressed-transcript set, enumerate
  every codon position. Reuse `tiling.group_into_tiles` (already pure geometry) and
  `extract.py`'s fetch+embed path unchanged.
- **Train/eval split of the codon alphabet:** train only on AUG + 9 near-cognate codons
  (as today). **Non-cognate codons are evaluation-only** — never trained — so "non-cognate
  ≈ 0" is a real out-of-distribution generalization test, not a memorized class.
- **Metrics:** recall @ FP/transcript on near-cognate decoys *at true (dense) imbalance*
  (reusing `caller.py`); mean p on held-out non-cognate codons → grounding check.

### Unresolved dependency (resolve before Phase 2 implementation)

Enumerating codons inside a transcript needs the **transcript's exon structure / mRNA
model** (to walk spliced coordinates and respect introns). The current manifest has
`transcript_id`, `gstart`, `mrna_index` but not full transcript models. Options to settle
in the Phase 2 spec: (a) swissisoform emits transcript sequences/exon coords; (b) tisiago
reads a GTF/annotation; (c) approximate via the genomic span between a transcript's known
candidates. This is the gating decision for Phase 2 and the reason it is staged second.

---

## Phase 3 — Narrow (after the dense scan works)

Make the statistics decisions once there's a working general scan: cross-validation
(replace single seed/split), seed variance, the true negative frequency to report at, and
which codon strata (esp. dTIS) to firm up. Settle the modeling that holds. Out of scope
for the first plans.

---

## Architecture impact

- **New:** `src/tisiago/caller.py` (Phase 1), `src/tisiago/scan.py` (Phase 2).
- **Unchanged / reused:** `tiling.py` (pure geometry — already reusable for scan),
  `extract.py` (fetch+embed path), `store.py` (row-aligned arrays).
- **Kept as PoC sanity checks:** `eval.py`, `resolution.py` — the ranking-track scripts;
  not deleted, not the deliverable.
- **Boundary preserved:** swissisoform owns *positives*; tisiago owns the *background* it
  scans and the *calibration/metrics*. gruyerenome still owns model internals.

## Risks

- Phase 1's headline number is on the curated 3:1 set until Phase 2 — must be labelled as
  such in any output, to avoid re-inflating the claim we just deflated.
- Calibrating on val (829 transcripts) then applying to test assumes cross-chromosome
  calibration transfers; the reliability curve on test is the check.
- Phase 2's enumeration dependency (transcript models) is unresolved and could pull in
  swissisoform — do not start Phase 2 until it's decided.
