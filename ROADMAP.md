# tisiago — roadmap & status

Single source of truth for the work. The north star (see `CLAUDE.md` → *Direction*):
a **general codon → P(initiation) predictor** — any codon in any expressed transcript →
a calibrated probability of being a translation-initiation site, including confidently
**rejecting non-starts**. Build broad first, then narrow.

Headline metrics: (1) curated-set discrimination judged **against the one-hot sequence
floor** (not chance); (2) near-neighbour win-rate @64bp (base resolution); and (3) recall at a
**validation-selected** FP/transcript threshold on the dense, true-imbalance test set.

_Last updated: 2026-09-09._

> **Current (protocol audit, 2026-09-09):** the dense experiment is complete, but its historical
> FP-budget thresholds were optimized on TEST and must be treated as oracle ranking diagnostics.
> The corrected pipeline separates fitting, calibration, operating-threshold selection, and final
> testing. The immediate task is a clean rerun, followed by frozen Evo2 context/stride/checkpoint
> and exon-spliced-input ablations. See `docs/INFERENCE_AND_EVALUATION.md`.

---

## Status at a glance

| Phase | What | Status | Artifacts |
|---|---|---|---|
| **PoC** | Frozen embeddings rank curated candidates | ✅ done | `eval.py`, `resolution.py`, `FINDINGS.md` |
| **P1** | Calibration machinery + caller metrics | ✅ **done, merged** | `caller.py` |
| **AR** | **Autoresearch fleet** (4 metrics) over the 1:3 set | ✅ **done + harvested 2026-06-16** | `autoresearch/winners.md`, `FINDINGS.md §6` |
| **P3** | Confirm winners across seeds/splits (was "autoresearch the head") | 🔄 **Leakage-free curated and dense multi-seed runners implemented; TXP extraction/evaluation chain in progress.** | `representation_eval.py`, `linear_head.py` |
| **P2 / Option B** | Imbalance-aware head @ true imbalance — **AG + Evo2** | 🟡 **Historical ranking result complete: imbalance-matched training wins and +Evo2 reaches oracle recall 0.300 / AUPRC 0.307. Single-seed; corrected fixed-threshold rerun required.** | `dense_caller.py`, `build_store.py`, SSD `dense_ag7` |
| ~~P2~~ | ~~Global all-codon dense *training*~~ | ⏸️ still deferred (Option B sidesteps it) | `enumerate_codons.py`, `scan_eval.py` |
| **P3.5** | **Head exploration — richer classifiers @ dense 2M** | 🟢 **DONE (2026-06-24): XGBoost 0.303 / LightGBM 0.304 AUPRC were within noise of logistic 0.307 in this single-seed comparison; no evidence that head capacity was limiting. §9.** | `head_xgb.py`, `compare_heads.py`, `FINDINGS.md §9` |
| **P4** | TIS efficiency regression (HeLa first) | 🟡 **first result (2026-06-24): Ridge on log1p(max_norm_HeLa) — in-dist ρ 0.30 / R² 0.24, but dense ranking collapses to AUPRC 0.098 (curated→dense shift, not absence of signal). §9.** | `efficiency_head.py`, `FINDINGS.md §9` |
| **P5** | Frozen-inference correctness + speed | 🟡 **TXP extractor and backend contract implemented. Evo2 0.6 / Vortex 1.1 is pinned and the deprecated API path is removed. Upgraded W8k features are bit-identical; TXP is 6.68x faster in model time on the matched shard. Curated and dense biological-accuracy gates are running.** | `extract_transcript.py`, `backend_contract.py`, `docs/INFERENCE_AND_EVALUATION.md` |

---

## P1 — calibration machinery ✅

`src/tisiago/caller.py`: train on `train`, isotonic-calibrate on held-out `val` (chr7),
report reliability + Brier + recall @ FP-per-transcript budget on `test`. Pure CPU, curated
3:1 store.

**Historical result:** AUPRC 0.741; isotonic barely moved Brier 0.1066→0.1064. The reported
**recall 0.733 @ ≤1 FP/transcript** used a TEST-optimized threshold and is therefore an oracle
ranking diagnostic. The corrected caller chooses the operating threshold on validation and
applies it unchanged to test. The curated set is also 3:1, not the deployment imbalance.

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

**LAUNCHED 2026-06-16.** Four git worktrees (`../tisiago-ar-{auprc,auroc,recall1fp,winrate64}`,
branches `ar/<obj>`, shared abs `STORE`), each running `autoresearch/launch_loop.sh` in a
detached tmux session (`ar-<obj>`). Each iteration: a fresh `claude -p` proposes ONE CONFIG
edit → the shell runs it on a CPU node (`srun --partition=20`) → keeps (commit) if
`<obj>_val` strictly beats the running best, else discards (`git checkout`). Per-objective
history lands in each worktree's `results.tsv` + git log; loop stdout in `autoresearch/loop.log`,
agent edits in `agent.log`.

**Monitor:** `tmux attach -t ar-<obj>` · `column -t -s$'\t' ../tisiago-ar-<obj>/autoresearch/results.tsv`
· `git -C ../tisiago-ar-<obj> log --oneline -10` · `squeue -u $USER`.
**Stop:** `tmux kill-session -t ar-<obj>` (finishes the in-flight iteration first).

---

## P2 / Option B — imbalance-aware head, evaluated at true imbalance 🟢 first result in (2026-06-18)

> **Result (2026-06-18, AG-only):** training **at** the true imbalance (49:1 dense) beats the
> 3:1-trained head decisively on the same TEST substrate — AUPRC 0.085→**0.246**, recall@≤1FP/tx
> 0.036→**0.225** (6×), non-cognate grounding 0.109→**0.0004**. The gains are rank-based, so this
> is the negative *distribution*, not just calibration. The §5 0.094 collapse was largely a
> train-prior artefact. See [`FINDINGS.md §7`](FINDINGS.md). Caveat: single seed, AG-only, Evo2
> (`ag7`) pending. This reframes the whole question — see the redesign decision doc (in progress).

Revives the dense scan **as an honest evaluation substrate**, not as training data — which is
what got the original P2 parked (it conflated the model with its train/eval negative
distribution). Option B keeps the autoresearch winner stack and trains on the **curated** set,
adding one imbalance-aware variant, then scores both at the real genome-wide imbalance:

- **Two heads** (`src/tisiago/dense_caller.py`), heavy L2 C=0.00075, isotonic-calibrated on
  curated val: **Config C** (`class_weight=None`, the AR best-all-rounder) vs **Option B**
  (`class_weight="balanced"`, set for ~230:1).
- **Evaluated on the dense scan TEST split** at true imbalance: recall @ ≤1 FP/tx over cognate
  (AUG+near_cognate) codons + non-cognate≈0 grounding + reliability/Brier — answering the
  FINDINGS §5 question (does imbalance-awareness lift the 0.094 collapse?).
- **3-gene out-of-sample demo** (SCN1A/GRIN1/TSC1): per-codon P(initiation) under both heads.

**Feature staging — AG first, Evo2 later** (`dense_caller.py --features {ag,ag7}`):
- **`ag` (near-term deliverable):** AG16k+AG131k+Kozak only — directly comparable to §5's
  AG-only 0.094, and ready as soon as the (fast) AlphaGenome scan finishes.
- **`ag7` (full):** adds Evo2 blk28 off{0,3,6,9} — the autoresearch winner stack — once the
  (slow) genome-wide Evo2 scan completes.

**Status (2026-06-17):**
- ✅ **AlphaGenome scan done** (AG16k + AG131k, 120/120 shards, A100) → AG store assembled
  (`scan_store_allsplits/`, both keys 62.7M×1536 fully covered) + Kozak staged.
- ✅ **First `--features ag` eval run** (AG+Kozak, no Evo2) at true 230.6:1 imbalance.
  **Sobering preliminary:** recall @ ≤1 FP/tx = **0.036** (Config C) / **0.050** (balanced);
  non-cognate grounding only partial (mean p ≈0.11, not ≈0); calibration doesn't transfer from
  3:1 (max_gap 0.77). Lower than §5's AG-only 0.094 — but not apples-to-apples (different
  feature set + `C`). **Caveat: AG-only, ≤1 FP/tx is the harshest metric, regularization tuned
  on 3:1.** A diagnostic re-run (recall-vs-budget *curve*, AUPRC, lighter `C`, saved preds) is
  in flight to separate *ranking-limited* from *brutal-operating-point* — §7 written after it.
- 🔄 **Evo2 blk28 background** on A6000 (0–59) + L40S (60–79), 4 GPUs, ~1.5 h/shard. The
  `--features ag7` re-run (adds base resolution — the component this metric most rewards) is the
  real test. See `HANDOFF_OPTION_B.md` (received plan; GPU/partition choices below supersede it).

**Robustness/throughput:** old evo2 OOM (`--mem=64G`) avoided by the **blk28-only config**
(`tis_evo2_8k_blk28.yaml`, 4 keys not 12) at `--mem=192G` (verified, no OOM). Evo2 sped up by
spanning A6000+L40S (both fit the 7B; A4000/t4 too small). Dense *training* stays deferred; the
all-splits store pre-stages it.

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

**Protocol and inference audit (2026-09-09).** Richer classifiers (XGBoost, LightGBM) at the true
2M imbalance both land within noise of the logistic head (AUPRC 0.303–0.307, recall@1FP
0.300–0.312). This is evidence that classifier capacity was not limiting in the tested setting,
not proof of an absolute feature ceiling. The project remains frozen-GLM inference: the next lever
is a better and cheaper inference representation, not a richer downstream head or
foundation-model training. See
`docs/INFERENCE_AND_EVALUATION.md`.

**Two open items, in priority order:**
1. **Protocol rerun + multi-seed confirmation.** Refit on unique sites, split validation
   transcripts into calibration and operating-threshold subsets, and report the untouched test
   result across SEED∈{0..4} with transcript-bootstrap confidence intervals. Historical
   recall-at-budget values used TEST-optimized thresholds and remain oracle ranking diagnostics.
2. **Frozen-inference Pareto sweep.** Run the implemented namespaced `W8kS4k`, `W4k`, and
   exon-spliced `TXP` arms against established `W8k/S2k`. First run `backend_contract`, then
   compare accuracy and end-to-end GPU time. The opt-in kernels helped fixed-length W8k but were
   much slower for variable-length TXP and remain an accuracy-gated ablation. True Evo2 batching
   belongs in `gruyerenome`; test the 1B checkpoint only as a separately namespaced representation.
