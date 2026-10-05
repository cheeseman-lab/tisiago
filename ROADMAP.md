# tisiago — roadmap & status

Single source of truth for the work. The north star (see `CLAUDE.md` → *Direction*):
a **general codon → P(initiation) predictor** — any codon in any expressed transcript →
a calibrated probability of being a translation-initiation site, including confidently
**rejecting non-starts**. Build broad first, then narrow.

Headline metrics: (1) curated-set discrimination judged **against the one-hot sequence
floor** (not chance); (2) near-neighbour win-rate @64bp (base resolution); and (3) recall at a
**validation-selected** FP/transcript threshold on the dense, true-imbalance test set.

_Last updated: 2026-10-05._

> **Current (2026-10-05): new direction — seq2func on raw Ribo-seq tracks (P6).** The frozen
> embedding + binary head line is closed at ≈0.30 recall @ ≈1 FP/transcript (clean protocol,
> FINDINGS §10): heads don't help (§9), and a sparse binary target on frozen features is the
> limiting design. Next: fine-tune AlphaGenome to predict raw Ribo-seq tracks across 5 cell lines
> × 2 replicates, then derive start-site calls on top. The frozen `ag7` head is the baseline.

---

## Status at a glance

| Phase | What | Status | Artifacts |
|---|---|---|---|
| **PoC** | Frozen embeddings rank curated candidates | ✅ done | `eval.py`, `resolution.py`, `FINDINGS.md` |
| **P1** | Calibration machinery + caller metrics | ✅ **done, merged** | `caller.py` |
| **AR** | **Autoresearch fleet** (4 metrics) over the 1:3 set | ✅ **done + harvested 2026-06-16** | `autoresearch/winners.md`, `FINDINGS.md §6` |
| **P3** | Confirm winners across seeds/splits | ✅ **Clean 5-seed dense rerun (2026-09-10): ensemble AUPRC 0.304 [0.288, 0.322], recall 0.306 @ 0.99 FP/tx with a val-selected threshold. 500k-negative cap (not yet matched to §7's 2M). §10.** | `representation_eval.py`, `linear_head.py` |
| **P2 / Option B** | Imbalance-aware head @ true imbalance — **AG + Evo2** | ✅ **Imbalance-matched training wins; +Evo2 reaches 0.300 / AUPRC 0.307 (single-seed, oracle threshold) — confirmed under the clean protocol in §10.** | `dense_caller.py`, `build_store.py`, SSD `dense_ag7` |
| ~~P2~~ | ~~Global all-codon dense *training*~~ | ⏸️ still deferred (Option B sidesteps it) | `enumerate_codons.py`, `scan_eval.py` |
| **P3.5** | **Head exploration — richer classifiers @ dense 2M** | 🟢 **DONE (2026-06-24): XGBoost 0.303 / LightGBM 0.304 AUPRC were within noise of logistic 0.307 in this single-seed comparison; no evidence that head capacity was limiting. §9.** | `head_xgb.py`, `compare_heads.py`, `FINDINGS.md §9` |
| **P4** | TIS efficiency regression (HeLa first) | 🟡 **first result (2026-06-24): Ridge on log1p(max_norm_HeLa) — in-dist ρ 0.30 / R² 0.24, but dense ranking collapses to AUPRC 0.098 (curated→dense shift, not absence of signal). §9.** | `efficiency_head.py`, `FINDINGS.md §9` |
| **P5** | Frozen-inference correctness + speed | ✅ **TXP extractor + backend contract; Evo2 0.6 / Vortex 1.1 pinned. TXP ≈ W8k on dense accuracy (AUPRC 0.299 vs 0.304, CIs overlap) at ~6.7× less Evo2 model time. §10.** | `extract_transcript.py`, `backend_contract.py`, `docs/INFERENCE_AND_EVALUATION.md` |
| **P6** | **Seq2func: fine-tune AlphaGenome on raw Ribo-seq tracks, call TIS on top** | ⬜ **next — needs a spec** | — |

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

## AR — autoresearch fleet ✅ (done, harvested 2026-06-16)

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

## P4 — TIS efficiency regression (HeLa first) 🟡 (first result, FINDINGS §9)

Move beyond yes/no into quantitative initiation: regress the unused per-condition
translational-efficiency label `max_norm_HeLa` on the frozen embeddings, restricted to
HeLa.

Ridge on `log1p(max_norm_HeLa)`: in-distribution ρ 0.30 / R² 0.24, but dense ranking collapses
to AUPRC 0.098 (curated→dense shift). Superseded by P6, which predicts the raw per-cell-line
tracks directly instead of regressing a summary label on frozen features.

---

## P6 — seq2func on raw Ribo-seq tracks ⬜ (needs a spec)

Fine-tune AlphaGenome to predict the raw Ribo-seq coverage tracks (5 cell lines × 2 replicates)
rather than training a binary head on sparse, frozen start-site labels. Start-site calls are then
derived from the predicted tracks.

**Open questions for the spec:**
- One model per cell line vs one multi-output model. The working hypothesis is that cross-line
  differences are mainly driven by which genes are transcribed — an unexpressed gene has a blank
  Ribo-seq track, which is missing data, not zero initiation.
- How expression enters: mask or condition on RNA abundance per line, and report the
  sequence-only model separately from any model that consumes measured expression.
- Where fine-tuning code lives (gruyerenome owns model loading/forward passes today).
- How to derive calls from predicted tracks, and how to score them against the frozen `ag7`
  baseline under the §10 protocol (recall @ val-selected FP/transcript, AUG vs near-cognate).
- Later: in-silico perturbation of the trained model.

## Immediate next action

1. **Spec P6** (brainstorm → spec → plan): data/track preparation per cell line × replicate,
   model/fine-tune setup, call derivation, and the evaluation contract against the frozen baseline.
2. **Carry the frozen baseline forward** under the §10 protocol so P6 has a fixed comparator,
   stratified by AUG vs near-cognate (near-cognate recall is 0.07 today).
