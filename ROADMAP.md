# tisiago — roadmap & status

Single source of truth for the work. The north star (see `CLAUDE.md` → *Direction*):
a **general codon → P(initiation) predictor** — any codon in any expressed transcript →
a calibrated probability of being a translation-initiation site, including confidently
**rejecting non-starts**. Build broad first, then narrow.

Headline metrics: (1) curated-set discrimination judged **against the one-hot sequence
floor** (not chance); (2) near-neighbour win-rate @64bp (base resolution). Deferred:
true-imbalance recall + non-cognate≈0 (the dense scan, parked).

_Last updated: 2026-06-16._

> **Reframe (current):** the full-genome dense scan (P2) is **parked** — it tangled the
> model with its train/eval negative distribution. We refocused on the **balanced 1:3
> curated set**, whose codon-matched negatives are already the rigorous control we want, and
> stood up a **4-way parallel autoresearch fleet** over the existing AG+Evo2 (+one-hot)
> features. See `docs/superpowers/specs/` and the plan `modular-splashing-sunset`.

---

## Status at a glance

| Phase | What | Status | Artifacts |
|---|---|---|---|
| **PoC** | Frozen embeddings rank curated candidates | ✅ done | `eval.py`, `resolution.py`, `FINDINGS.md` |
| **P1** | Calibration machinery + caller metrics | ✅ **done, merged** | `caller.py` |
| **AR** | **Autoresearch fleet** (4 metrics) over the 1:3 set | 🟢 **harness built + baselines; run paused for go** | `autoresearch/` |
| **P2** | Global all-codon dense caller | ⏸️ **parked** (code merged, deferred) | `enumerate_codons.py`, `scan_eval.py` |
| **P4** | TIS efficiency regression (HeLa first) | ⬜ not specced | — |

---

## P1 — calibration machinery ✅

`src/tisiago/caller.py`: train on `train`, isotonic-calibrate on held-out `val` (chr7),
report reliability + Brier + recall @ FP-per-transcript budget on `test`. Pure CPU, curated
3:1 store.

**Result:** AUPRC 0.741; the logistic head is *already well-calibrated* (isotonic barely
moves Brier 0.1066→0.1064); **recall 0.733 @ ≤1 FP/transcript**. Because isotonic is
monotonic the recall is calibration-invariant. Honest caveat baked into the output: curated
3:1, not true imbalance.

---

## AR — autoresearch fleet 🟢 (active, run paused)

Stood up `autoresearch/` — a **4-way parallel fleet**, each loop climbing a different
objective on **val**, reporting **test** (never selecting on test):

| objective | what | baseline (test) |
|---|---|---|
| `auprc` | precision-aware ranking | 0.76 |
| `auroc` | overall ranking | 0.90 |
| `recall1fp` | caller operating point | 0.75 |
| `winrate64` | hard base-resolution discrimination | 0.82 |

**Levers swept** (edit `train_experiment.py` CONFIG): feature subset (14 embedding keys +
2 one-hot), head (logistic/MLP), regularization, `class_weight`, train-negative subsample.
Fixed metric harness in `evaluate.py`; results logged per-worktree in `results.tsv`.

**One-hot grounding (your call — added & paid off):** `make_onehot.py` →
`onehot/{codon12,kozakW20}.npy`. Establishes the floors every config is judged against:
- one-hot **codon** → AUROC **0.49** (≈chance): control passes — signal is contextual, not
  codon identity.
- one-hot **±20bp sequence** → AUROC **0.75**, win@64 **0.73**: the real floor. Foundation
  lift is **0.75→0.90**, not 0.5→0.90.

**Harness verified:** reproduces FINDINGS (AG16k+Evo2 → 0.90/0.76/0.82). Baselines seeded in
`autoresearch/results.tsv`.

**Next (paused for go):** create 4 git worktrees (one per objective, shared abs `--store`),
launch the `autoresearch` skill in each → overnight keep/discard loops.

---

## P2 — global all-codon dense caller ⏸️ parked

Code merged and working (`enumerate_codons.py`, `scan_eval.py`, `run_tis_scan.sh`), GPU jobs
cancelled. Parked because it conflated the model with its train/eval negative distribution
(see Reframe). The one real result stands as FINDINGS §5: **AG-only recall 0.094 @ ≤1
FP/transcript at true 230:1 imbalance**, non-cognate grounding only partial (mean p 0.12).
**Architectural debt if revived:** `extract.py` accumulates sliced vectors in RAM → dense
evo2 needs `--mem=384G` (OOM-killed at 64G); should stream parts to disk.

---

## P4 — TIS efficiency regression (HeLa first) ⬜ (not specced)

Move beyond yes/no into quantitative initiation: regress the unused per-condition
translational-efficiency label `max_norm_HeLa` on the frozen embeddings, restricted to
HeLa.

**Open questions before a spec:** which rows (positives only? expressed_HeLa filter?);
target transform (raw `max_norm_HeLa` vs log); metric (Spearman / R²); whether to reuse the
curated store or the dense scan; how the multi-line labels (K562/RPE1/U2OS) factor later.

---

## Immediate next action

**Paused before launch.** The autoresearch harness is built, verified, and baseline-seeded.
On go: create the 4 worktrees and start the `autoresearch` skill in each (CPU, overnight).
No GPU. Then read `results.tsv` across the fleet to see which features/heads each metric
favors — especially whether Evo2 dominates `winrate64` and what beats the 0.75 sequence floor.
