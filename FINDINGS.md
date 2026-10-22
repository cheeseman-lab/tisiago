# tisiago — findings

Proof-of-concept results for predicting translation-initiation sites (TIS) from
**frozen** genome-foundation-model embeddings with lightweight downstream heads. All
numbers are on **held-out chromosomes** (test = chr8/chr9, val = chr7) with a simple
standardized-logistic head — no fine-tuning, a track no foundation model is trained
to emit.

**Setup.** 192,072 candidate codons (48,018 called TIS = positives + 144,054 matched
in-transcript negatives at 3×; 24.8% positive on the test split). Negatives are
uncalled near-cognate/ATG codons in the *same* transcripts as positives, so the task
is genuine codon-vs-codon discrimination, not "coding region vs not" — and the
host-transcript expression is shared, removing the expression confound. Head: logistic
regression on standardized features, trained on a 60k subsample of the train
chromosomes. Reproduced identically after the repo migration (`tisiago.eval`).

## 1. Which embeddings carry TIS signal

| Feature set | dim | AUROC | AUPRC |
|---|---|---|---|
| AlphaGenome 16k | 1536 | 0.844 | 0.68 |
| AlphaGenome 131k | 1536 | 0.858 | 0.69 |
| Evo2 blk28 | 4096 | 0.881 | 0.70 |
| **AlphaGenome 16k + Evo2 blk28** | 5632 | **0.901** | **0.75** |

- A **linear** head suffices — a 256-unit MLP on the combined set does not beat logistic
  (0.899 vs 0.901), i.e. the signal is linearly decodable.
- **Evo2 alone > AlphaGenome alone**; combining them helps → they carry complementary
  signal (concatenate). AUPRC baseline (positive rate) = 0.248.

## 2. Is it the *interesting* kind of signal? (stratified by start type)

`AUROC` within each stratum = "can the head rank these positives above the negatives".
The novel question is whether **non-canonical / alternative** TIS still separate (not
just the canonical starts AlphaGenome already "knows" from gene annotation).

| Stratum | n | AUROC |
|---|---|---|
| ALL positives (headline) | 3,653 | 0.901 |
| canonical annotated starts (easy) | 1,325 | 0.927 |
| **NON-canonical (alternative TIS)** | **2,328** | **0.886** |
| ↳ uORF (5′UTR) | 584 | 0.947 |
| ↳ N-terminal extension | 495 | 0.936 |
| ↳ downstream in-CDS (dTIS) | 1,201 | 0.840 |

**The signal is not a canonical-start artifact** — it barely drops when canonical starts
are removed (0.927 → 0.886). uORFs and N-terminal extensions separate at ~0.94; the
hardest case (downstream alternative starts) is still 0.84.

## 3. Is it at true single-nucleotide resolution?

`win@Dbp` = P(a real TIS scored above a decoy candidate codon within `D` bp in the
**same transcript**). Long-range/regional context is useless here, so this isolates
genuine base-resolution discrimination.

| Model | win@64bp | win@128bp | win@512bp | win@2000bp |
|---|---|---|---|---|
| AlphaGenome 16k | 0.698 | 0.714 | 0.748 | 0.785 |
| Evo2 blk28 | 0.754 | 0.755 | 0.779 | 0.814 |
| **AlphaGenome + Evo2** | **0.815** | 0.814 | 0.828 | 0.858 |

- **There is genuine base-resolution signal** — even against an immediately adjacent
  decoy (≤64 bp), the combined head wins 0.82 of the time (≫ 0.5).
- **But weaker than the headline AUROC.** Much of the 0.84–0.90 global AUROC is
  regional context (decoys 2 kb away are easy). The honest "can you pinpoint the exact
  start codon" number is ~0.70–0.82.
- **Evo2 (1 token = 1 bp) carries the nucleotide resolution; AlphaGenome (128 bp-
  upsampled "1bp" decoder) carries regional context.** Evo2's edge over AlphaGenome is
  largest exactly at 64 bp (+0.056 vs +0.029 at 2 kb), as expected from the architectures.

## 4. One-hot grounding — the floor the embeddings actually beat

The headline AUROC was being read against an implicit chance baseline (0.5). Two one-hot
controls (logistic head, same splits) reset that baseline — the real question is what the
foundation embeddings beat:

| Features | dim | AUROC | AUPRC | win@64bp |
|---|---|---|---|---|
| one-hot **codon** (control) | 12 | **0.490** | 0.243 | 0.450 |
| one-hot **±20 bp sequence** (floor) | 164 | **0.753** | 0.530 | 0.728 |
| AlphaGenome 16k + Evo2 blk28 | 5632 | 0.905 | 0.760 | 0.816 |

- **The codon control is ≈ chance (0.49).** Codon identity alone carries no signal — because
  negatives are codon-frequency-matched to positives. This is hard proof the embeddings'
  signal is **contextual, not "is it an AUG."**
- **The sequence floor is high (0.75 / 0.73).** A trivial one-hot logistic on ±20 bp of raw
  sequence (Kozak context) already reaches 0.75 AUROC. So the foundation models' honest lift
  is **0.75 → 0.90 AUROC and 0.73 → 0.82 win@64bp**, not 0.5 → 0.90. The embeddings clearly
  win — most at base resolution — but the marginal value over raw local sequence is the
  number to quote, not the raw 0.90.

(Baselines live in `autoresearch/results.tsv`; one-hot arrays at
`data/store/embeddings/onehot/{codon12,kozakW20}.npy`.)

## 5. At true imbalance — the honest caller number (Phase 2, **parked, AG-only**)

Sections 1–3 are on the curated 3:1 decoy set. Phase 2 scored *every codon* in the
held-out transcripts (dense scan, GENCODE v49) and applied the calibrated head at the
**realistic** imbalance. This direction is now **parked** (it conflated the model with its
train/eval negative distribution — see ROADMAP); the one AG-only result below stands:

| Metric | curated 3:1 | **true imbalance (230:1)** |
|---|---|---|
| neg:pos per transcript | 3:1 | **230.6:1** (~570 candidate codons/transcript) |
| recall @ ≤1 FP/transcript (AG-only) | — | **0.094** (p≥0.885) |
| non-cognate mean p / p95 / FPR@thr | — | 0.119 / 0.483 / 0.0016 |

**Why the headline collapses from the 0.9 you remember.** Those are *different metrics*,
increasingly honest about the actual task — not the same metric degrading:

- **AUROC 0.90** (§1) — ranking, *imbalance-blind*: "tell a TIS from a decoy". Flattering.
- **AUPRC 0.75** (§1) / **recall@budget 0.73** — precision-aware, still on the 3:1 set.
- **recall@budget 0.094** — the *same* operating-point metric once negatives are realistic
  (230:1). "Call the start among ~570 candidate codons."

Each step measures a harder, realer question; AUROC was never wrong, it just answers the
easy one. This is exactly the calibrated-imbalance number the §Caveats called for.

> **Update (2026-06-18): the collapse is largely a *train-prior* artefact, not a ceiling.**
> §7 below trains the head **at** the true imbalance and recovers most of the recall **and**
> fixes the grounding — the 0.094 here is the 3:1-trained head testing out-of-distribution.

**Grounding is only partial for AG alone.** Non-cognate codons (never trained on) are *not*
called (FPR 0.0016 at the operating threshold) but are *not* crushed to ≈0 either (mean p
0.12, p95 0.48) — AlphaGenome's 128 bp-upsampled embedding leaks regional probability onto
non-cognate positions. Evo2 (1 token/bp, true codon identity) is expected to lift both the
recall and the grounding — but the dense direction is parked pending the autoresearch pass.

## 6. Autoresearch — best head per objective (2026-06-16)

Four parallel autoresearch loops, each climbing one metric on **val** and reporting **test**
(never selecting on test), swept feature subsets × head × regularization × class-weighting on
the curated 1:3 set (~20 experiments/loop to plateau). Every objective beat the 2-key baseline
**on test**, in lockstep with val — the gains are real, not val hill-climbing:

| metric | baseline (test) | **best (test)** | winning config |
|---|---|---|---|
| AUPRC | 0.760 | **0.797** | 7-key stack, C=0.00075, no weight |
| AUROC | 0.905 | **0.920** | 7-key stack, C=0.002, balanced |
| recall@≤1FP | 0.751 | **0.801** | 7-key stack, C=0.003, balanced |
| win@64bp | 0.816 | **0.834** | 7-key stack + codon one-hot, C=0.1, balanced |

The **7-key stack** all four converged on: `AG16k + AG131k + Evo2 blk28 off{0,3,6,9} +
Kozak one-hot` (19.6k-dim). Key results (full detail + cross-metric matrix in
[`autoresearch/winners.md`](autoresearch/winners.md)):

- **A richer feature stack helps every metric.** Stacking AG131k (regional) + 3 extra Evo2
  offsets (resolution) + explicit Kozak each added signal over the 2-key headline — embeddings
  and explicit local sequence are complementary.
- **One config is the best all-rounder**: 7-key, heavy L2 (C=0.00075), *no* class weight —
  tops auprc, auroc **and** recall@1FP on test at once. `balanced` helped val but slightly
  val-overfits relative to test.
- **Precision ↔ resolution tension.** Maxing win@64 needed explicit codon identity + loose L2,
  which *costs* the precision metrics. The resolution-optimal head ≠ the precision-optimal head.
- **Linear still suffices** — every MLP lost or crashed; all four winners are logistic.

These are **single-seed point estimates** (best-of-search). Phase 3 is to confirm them across
seeds/splits before any figure.

## 7. Imbalance-matched training lifts the collapse (2026-06-18, AG-only)

§5 applied a **3:1-trained** head at 230:1 and recall fell to ~0.04–0.09. The obvious
suspect was train/test prior mismatch. We tested it directly: same AG-only 3-key stack
(`AG16k + AG131k + Kozak`), same held-out TEST substrate (1.82M codons: 823k cognate +
1M never-trained non-cognate, 3,555 positives, true 230.6:1), only the **training prior**
changed — `curated-C` trains on the 3:1 set; `Dense` trains on the dense pool at **49:1**
(all train positives + 2M capped negatives), `bal` = `class_weight="balanced"`, `none` = no
reweight. All scored at the same true imbalance:

| metric (TEST @ 230:1) | curated-C (train 3:1) | Dense bal (train 49:1) | **Dense none (train 49:1)** |
|---|---|---|---|
| AUPRC (base 0.0043) | 0.0845 | 0.2023 | **0.2456** |
| recall @ ≤1 FP/tx | 0.036 | 0.193 | **0.225** |
| recall @ ≤5 FP/tx | 0.259 | 0.437 | **0.473** |
| recall @ ≤20 FP/tx | 0.533 | 0.682 | **0.688** |
| non-cognate mean p (→0) | 0.109 | 0.0004 | **0.0003** |
| non-cognate p95 | 0.476 | 0.0012 | **0.0008** |
| cognate Brier | 0.039 | 0.0038 | **0.0036** |

**What it proves — the gain is in the *ranking*, not just the threshold.**

- **AUPRC tripled** (0.085 → 0.246) and **recall@≤1FP/tx went 6×** (0.036 → 0.225). AUPRC is
  rank-based and calibration-invariant — **no recalibration of the 3:1 head could produce
  this.** Training against the genome's *diverse* negatives (not just matched near-cognate
  decoys) taught a genuinely better boundary. The negative *distribution* matters.
- **The grounding check flips from fail to pass.** Non-cognate mean p: 0.109 (p95 0.476 —
  half of certain-negatives look like plausible starts) → **0.0004**. The imbalance-trained
  head drives "non-cognate ≈ 0" — but **§8 shows this absolute level is largely a *calibration*
  property** (recalibrating the 3:1 head alone also reaches ≈0.003), so it is not by itself
  proof of learned biology; the durable discriminator is the rank-based cognate metric.
- **At 49:1, no class-weighting wins** — `Dense(none)` ≥ `Dense(bal)` on every metric. Raw
  imbalance-matched training beats the `balanced` reweight.

**Evo2 lift — matched `ag` vs `ag7` pair at 2M (2026-06-23, clean SSD pipeline).** Adding the
Evo2 blk28 off{0,3,6,9} stack to AG+Kozak, both at the **matched 2M-negative** cap (49:1, the
same cap as the headline AG table above), best head `Dense(None)`, same TEST @ 230:1. Run
through the migrated SSD substrate (`build_store.py`); the `ag` column reproduces §7's original
2M AG **bit-for-bit** (AUPRC 0.2456, recall 0.225), validating the migration:

| metric (TEST @ 230:1) | `ag` (2M) | **`ag7` (2M, +Evo2)** |
|---|---|---|
| AUPRC (base 0.0043) | 0.2456 | **0.3073** |
| recall @ ≤1 FP/tx | 0.225 | **0.300** |
| recall @ ≤5 FP/tx | 0.473 | **0.539** |
| recall @ ≤20 FP/tx | 0.688 | **0.769** |
| non-cognate mean p (→0) | 0.0003 | 0.0003 |
| cognate Brier | 0.0036 | 0.0034 |

The two levers do **different jobs**: imbalance-matched training already drove non-cognate to
≈0 (grounding, solved by AG alone), so Evo2's **+33% recall / +25% AUPRC** lift is **cognate
discrimination** — its base resolution helps rank a true start above near-cognate decoys. Best
caller to date: **`ag7` Dense(None), recall 0.300 @ ≤1FP/tx at true 230:1, AUPRC 0.307 (71×
base), non-cognate ≈0.0003.** The lift held at the earlier matched 1M cap too (0.247→0.309
recall), so it is not a cap artefact. (Even the 3:1-trained `curated-C` improves with Evo2 —
recall 0.036→0.142 — but stays far below imbalance-matched training.)

**Caveats (do not overclaim).** **Single seed** is the one remaining caveat — the headline AG
and the Evo2-lift pair are now at the *same* 2M cap (the earlier cap mismatch is resolved), and
the *balanced-train + recalibrate* baseline has been run (§8: it does not reach dense). The
AUPRC/recall gains are rank-based, so recalibration alone cannot close them. Confirm across
seeds before any figure.

## 8. Negative control — recalibration recovers calibration, not ranking (2026-06-22)

§7 argued the dense-training gain is rank-based and so *cannot* be a calibration artefact. The
direct test (operon's predicted control): take the curated-trained head's saved test
predictions and apply the **Saerens prior-correction** (a monotonic map from the curated 3:1
prior to the true cognate prior 0.0043), then re-measure. Monotonic ⇒ ranking is mathematically
untouched; only calibration can move (`scripts/saerens_control.py`):

| `ag7`, cognate test @ 230:1 | curated-C raw | curated-C **+Saerens** | Dense(None) |
|---|---|---|---|
| AUPRC (ranking) | 0.1405 | **0.1405** | **0.2959** |
| Brier (calibration) | 0.0437 | **0.0040** | 0.0035 |
| non-cognate mean p | 0.1138 | **0.0032** | 0.0004 |

(`ag` is the same story: AUPRC 0.0845→0.0845, Brier 0.0387→0.0041, non-cog 0.109→0.003.)

- **Recalibration fully recovers calibration** (Brier 0.044→0.004, ≈ dense) but leaves AUPRC
  *exactly* unchanged (0.1405 vs dense 0.296). No recalibration of the curated head can reach
  the dense head's ranking → **the §7 lift is a negative-*distribution* effect, not a prior
  shift.** D1 settled (Saerens 2002; Dal Pozzolo 2015; see the decision doc).
- **Refinement — grounding is partly a calibration property.** Non-cognate mean p drops
  0.114→0.003 under recalibration *alone*. So "non-cognate ≈ 0" is **not by itself** proof of
  learned initiation biology — a recalibrated curated head also achieves it. The honest
  discriminator of "learned biology" is the **rank-based cognate metric** (AUPRC / recall@FP),
  which only the right negative *distribution* improves. Read §7's grounding flip with this
  caveat: the *absolute* non-cognate level is recoverable post-hoc; the *ranking* is not.
- **Balanced-*training* doesn't help either (2026-06-22).** Operon's literal arm — train the
  curated head with `class_weight="balanced"` then Saerens-recalibrate — was run
  (`dense_caller --train curated`). Balanced curated training is *slightly worse* than
  unweighted, not better: AUPRC ag7 **0.123** (bal) vs 0.141 (unweighted) vs **0.296** (dense);
  ag 0.078 vs 0.085 vs 0.242. Saerens (monotonic) then leaves AUPRC untouched. So **no curated
  head — weighted or not, recalibrated or not — reaches dense.** Reweighting during training
  cannot substitute for the right negative *distribution*; this exhaustively closes D1.

## 9. Richer tested heads don't beat logistic (2026-06-24)

§7 set the dense-trained logistic head at **recall 0.300 @ ≤1 FP/tx, AUPRC 0.307**. The natural
next question: is the *head* the bottleneck? Phase 3 swaps the classifier while holding everything
else fixed — same 7-key `ag7` stack (19.6k-dim), same dense TRAIN split at 49:1, same dense TEST
at 230:1, same metric harness (`scripts/compare_heads.py` recomputes every head through the
*identical* `recall_at_fp_budget`/`reliability`/`grounding_stats`, so the comparison is
apples-to-apples). Gradient-boosted trees (`src/tisiago/head_xgb.py`), isotonic-calibrated on
dense val:

| `ag7`, cognate test @ 230:1 | **Logistic** (§7) | **LightGBM** | **XGBoost** |
|---|---|---|---|
| AUPRC (base 0.0043) | **0.307** | 0.304 | 0.303 |
| recall @ ≤1 FP/tx | 0.300 | **0.312** | 0.309 |
| recall @ ≤5 FP/tx | **0.539** | 0.507 | 0.504 |
| recall @ ≤20 FP/tx | **0.769** | 0.738 | 0.762 |
| non-cognate mean p (→0) | 0.0003 | 0.0017 | 0.0017 |
| cognate Brier | 0.0034 | 0.0035 | 0.0035 |

- **No head meaningfully beats logistic.** Three distinct model classes — linear logistic,
  leaf-wise boosting (LightGBM), level-wise boosting (XGBoost) — cluster inside noise (AUPRC
  0.303–0.307, recall@1FP 0.300–0.312) at matched 2M scale. Trees nudge the *tightest* operating
  point up ~1 pt (0.31 vs 0.30) but give it back in the mid-range (AUPRC and recall@5FP both
  lower). Nonlinear interaction-learning extracts nothing extra from these frozen-embedding
  features that a linear boundary misses. This single-seed comparison provides no evidence that
  classifier capacity is limiting; it does not establish an absolute feature ceiling. The next
  lever is the representation itself, not a fancier head.
  Grounding holds across heads (non-cognate ≈ 0.002, both ≈ 0).
- **Phase 4 — continuous efficiency regression (`src/tisiago/efficiency_head.py`).** A Ridge head
  on `log1p(max_norm_HeLa)` predicts efficiency **in-distribution** on curated val at **R² 0.24,
  Spearman ρ 0.30** (best at heavy α=100; the 19.6k features are badly collinear — rcond ~5e-9).
  So efficiency *is* weakly decodable. But scored as a ranker on the dense TEST split it collapses
  to **AUPRC 0.098 / recall@1FP 0.141** — landing exactly where every *other* curated-trained head
  lands on dense (curated-C 0.141, §5 AG-only 0.094). That collapse is the **train-curated /
  eval-dense shift** (§5), not absence of signal: reframing the target as continuous efficiency
  did **not** rescue the distribution-shift problem. (Brier/grounding are N/A for a raw regressor,
  so Ridge sits on its own normalized eval surface, not the classifier table above.)

## 10. Clean-protocol rerun + exon-spliced Evo2 (2026-09-10)

Rerun of the §7 `ag7` dense head under the leakage-free protocol
(`docs/INFERENCE_AND_EVALUATION.md`): unique sites only; validation transcripts split into a
calibration half and an operating-threshold half; threshold chosen on operating validation and
applied once to TEST; 5 negative-subsample seeds (500k train negatives each, not §7's 2M);
transcript-bootstrap 95% CIs (1000 replicates). TEST = 823,302 cognate codons at true imbalance.
Two arms differ only in the Evo2 input: genomic 8 kb windows (`W8k`) vs the complete exon-spliced
mature transcript (`TXP`). Run: `scripts/run_tis_dense_representation_eval.sh`, outputs in
`data/dense_txp_representation/`.

| 5-seed ensemble, TEST @ true imbalance | `AG + W8k` | `AG + TXP` |
|---|---|---|
| AUPRC | 0.304 [0.288, 0.322] | 0.299 [0.281, 0.316] |
| recall @ val-selected threshold | 0.306 [0.291, 0.322] | 0.316 [0.301, 0.331] |
| precision | 0.433 | 0.423 |
| FP / transcript (budget 1.0) | 0.99 | 1.06 |
| win@64bp | 0.883 | 0.877 |

Per-seed AUPRC 0.284–0.294 (`W8k`) and 0.279–0.287 (`TXP`). Paired TXP − W8k differences:
AUPRC [−0.013, +0.002], recall [+0.001, +0.019], win@64 [−0.009, −0.003].

- **The §7 headline survives an honest protocol.** Recall ≈0.30 at ≈1 FP/transcript now uses a
  validation-chosen threshold, so it is a deployable operating point rather than an oracle one.
- **TXP ≈ W8k on accuracy at ~1/7 the Evo2 model time.** AUPRC is indistinguishable; win@64 is
  slightly lower. No non-inferiority margin was fixed before the run, so this is not yet a formal
  acceptance.
- **The caller is mostly an AUG caller.** Near-cognate starts are rarely recovered:

| `AG + W8k` ensemble stratum | AUPRC | recall | precision |
|---|---|---|---|
| AUG | 0.514 | 0.577 | 0.532 |
| near-cognate | 0.087 | 0.068 | 0.181 |
| plus strand | 0.324 | 0.318 | 0.460 |
| minus strand | 0.285 | 0.294 | 0.407 |

  (`TXP` is the same picture: AUG recall 0.600, near-cognate 0.066.) The minus-strand deficit is
  consistent across both arms and is not yet explained.

**Caveat.** 500k train negatives vs §7's 2M — the clean rerun is not yet cap-matched to §7.

**Deliverable — discrete calls.** `python -m tisiago.call_starts` (Slurm:
`scripts/run_tis_call_starts.sh`) applies the 5-seed `AG + W8k` ensemble and its validation-chosen
threshold (0.2318) to the held-out chr8/chr9 cognate codons and writes
`data/calls/ag_w8k_test_calls.parquet` (823,302 codons, 2,514 called) plus a summary JSON. The run
refuses to write unless it reproduces the logged ensemble metrics; it did (recall 0.306,
0.99 FP/transcript). One call differs from the logged evaluation: the evaluation scored with the
unfolded sklearn pipeline, the deployable artifacts with folded coefficients, and isotonic steps
turn a float-precision logit difference into a visible probability jump for 3 codons at the
threshold. Training-chromosome calls are not produced (the model saw them).

## 11. P6 — fine-tuning AlphaGenome on Ribo-seq tracks (in progress, 2026-10-06)

P6.1 overfit gate: can a model reproduce base-resolution Ribo-seq profiles on a single window?
Window: HeLa, chr12 131,072 bp centred on 6,536,000 (GAPDH), 6 channels (TIS/CHX/RNA x strand),
canonical P-site targets, mask all True. Profile KL = mean multinomial KL over channels (deviance
form). References: uniform prediction KL ~4.74; replicate floor (pooled vs rep2, smoothed 5 bp)
0.35-1.41, mean ~0.6. Gate: total-loss reduction >= 0.9 AND final profile KL <= 1.0 (the profile
gate was added after the first run met the total-loss gate by fitting window totals alone).

| run | steps | final KL | best KL | s/step | peak GiB |
|---|---|---|---|---|---|
| frozen backbone, MLP head, lr 1e-3 | 300 | 4.31 | - | 0.080 | 8.70 |
| frozen, MLP, lr 1e-3 | 3,000 | 3.83 | - | 0.080 | - |
| frozen, MLP, lr 3e-3 | 3,000 | 3.71 | - | 0.081 | - |
| frozen, conv head | 3,000 | 4.64 | - | 0.151 | 8.70 |
| fine-tuned backbone (lr 1e-5), MLP | 3,000 | 2.52 | 2.24 | 0.263 | 17.27 |
| fine-tuned backbone (lr 1e-5), conv | 3,000 | 4.55 | - | 0.359 | 18.46 |
| fine-tuned (1e-4), MLP, constant lr | 6,000 | 2.15 | 1.44 | 0.263 | 17.27 |
| fine-tuned (1e-4), MLP, cosine + warmup 300 + clip 1.0 | 6,000 | 1.005 | 1.005 | 0.265 | 16.96 |
| **fine-tuned (1e-4), MLP, cosine + warmup 300 + clip 1.0** | **12,000** | **0.762** | **0.762** | 0.265 | 16.93 |

All runs on one L40S. The frozen conv head sees embeddings only (agft never forwards sequence to
custom heads, `custom_model.py:2338`).

- **Frozen AG 1-bp embeddings cannot reproduce base-resolution profiles**, even when memorizing one
  window (KL 3.7-4.6 vs uniform 4.74). Consistent with §3: the 1-bp embeddings mostly carry
  regional context.
- **Fine-tuning the backbone does** (0.76, near the replicate floor), but needs warmup-cosine and
  gradient clipping to be stable (constant lr 1e-4 oscillated: best 1.44, final 2.15).
- The conv head was unstable at lr 1e-3 and is inconclusive.
- **One window, memorization, not generalization.** P6.2 tests held-out chromosomes.
- Alignment check: top TIS+ peaks sit on the GAPDH ATG (A 11,182 / T 16,720 counts; the +13 offset
  puts the max on the T); CHX downstream frame-0 fraction 0.59. P6.0 QC: replicate r 0.96-0.99;
  CHX frame-0 0.45-0.67 after the Ribo-TISH minus-strand +1 correction.
- The P6.0 round trip result is in §12 (fails 0.95 on all six conditions).

Deviations from the spec found by the spike: own training loop (agft 0.1.12 trainer is broken);
agft LoRA is head-side only; full fine-tuning fits at 131 kb in ~17 GiB on a 48 GB card (the spec's
>=76 GB was for 1 Mb inputs); A100 queue unavailable, so all runs used L40S.

## 12. P6 experiment 1 — per-condition track models (2026-10-10 to 2026-10-17)

The first full P6 experiment: one alphagenome-ft model per condition (6 channels: TIS/CHX/RNA x
strand, 1-bp, replicate-pooled targets, Poisson-multinomial loss, end-to-end fine-tuning), trained on
all chromosomes except chr7/8/9 and scored on val (chr7). Ledger:
`.superpowers/sdd/2026-10-10-p6-autoresearch/progress.md`; per-trial scores
`autoresearch/p6/results.tsv`. Run directories are kept under `data/p6/runs/` (list below).

**Data fix first.** K562/U2OS libraries 718-733 had been built from one sequencing run copied twice
(every read duplicated; the other run unused). Rebuilt from both runs (+56-66% unique reads);
canonical Ribo-TISH calls are now `ribotish/results/2026-10-09_reprocessed`. HeLa/RPE1 unaffected.

**Base model (the tutorial recipe).** Fits the tracks, but has no codon periodicity: CHX
dominant-frame agreement with the data 0.43 on K562 (chance 0.33; replicate vs replicate 0.87).

**Periodicity search (K562, ~25 trials).** What worked: a residual head on the base model that
multiplies only the CHX channels by a bounded, sequence- and transcript-feature-driven factor
(`resid_chx`), a codon-phase loss at weight 200, a gentle second stage (backbone lr 1e-5), 30k steps.
What did not: fresh heads (lose fit on every channel), second stage at full lr (loses fit even with
no new loss), an unbounded modulation (loss blow-up), higher modulation lr.

| condition | gene log-Pearson base -> winner | CHX frame agreement base -> winner | replicate |
|---|---|---|---|
| K562 | 0.904 -> 0.909 | 0.43 -> 0.89 | 0.87 |
| HeLa | 0.834 -> 0.839 | 0.35 -> 0.59 | 0.61 |
| U2OS | 0.877 -> 0.880 | 0.42 -> 0.64 | 0.74 |
| RPE1 Async | 0.936 -> 0.934 | 0.40 -> 0.52 | 0.61 |
| RPE1 Que | 0.904 -> 0.902 | 0.42 -> 0.69 | 0.46 |
| RPE1 Sen | 0.918 -> 0.920 | 0.39 -> 0.50 | 0.62 |

All six pass the fit gate (profile KL <= base + 0.02, gene log-Pearson >= base - 0.005). Predicted
codon amplitude stays too weak (count-weighted frame-0 share e.g. K562 0.52 vs 0.70 observed).

**RNA channels (base models).** Gene-level log-Pearson 0.83-0.94 vs replicate 0.94-0.99; RPE1
closest, HeLa furthest.

**Start sites from predicted tracks: negative.**
- Round trip (real tracks -> synthetic BAMs -> Ribo-TISH vs real-BAM calls): Jaccard 0.85-0.89 in
  all six conditions (recovers 95-97%, adds 8-12%); fails the 0.95 bar. Comparisons below use the
  same synthetic route for both arms.
- Ribo-TISH on predicted vs real K562 tracks (chr7, depth-matched): Jaccard 0.15 (annotated starts:
  precision 0.77-0.79, recall 0.41); same-pipeline resample of the real data reaches 0.66-0.67.
- Cause: predicted harringtonine peaks are at the right nucleotide but ~6x too flat (share of local
  signal at real starts 0.044 vs 0.252); modulating the TIS channels too raised it only to 0.055.
- On the frozen baseline's protocol (val operating half, recall @ ~1 FP/transcript): predicted tracks
  0.035 (all lines) / 0.099 (K562 only); real tracks 0.254; frozen `ag7` 0.306 (test). Test
  chromosomes not touched.

**Conclusion.** Fine-tuned AlphaGenome is a good Ribo-seq *track* model (fit, and with the residual
CHX head, replicate-level periodicity in K562), but reading start sites off its tracks does not beat
predicting labels directly — and even real tracks, scored that way, fall below the baseline.
Next (ROADMAP P6): condition on the cell line's RNA track (cheap to generate) and predict
harringtonine + CHX, evaluated leave-one-cell-line-out.

Kept runs (`data/p6/runs/`): `k562_base`, `ar_<c>_base` (hela, u2os, rpe1_async, rpe1_que, rpe1_sen),
`ar_resid_chx_w200_30k` (K562 winner) and `ar_<c>_resid_chx_w200_30k`, `ar_resid_chxtis_s{3,1.5}_w200_30k`;
round trip `data/p6/roundtrip_reproc/`; baseline-protocol eval `data/p6/track_tis_eval/val/`.

## 13. P6 experiment 2 — RNA-conditioned model, leave-one-cell-line-out (2026-10-19)

Spec: `docs/superpowers/specs/2026-10-19-p6-rna-conditioned-design.md`. One alphagenome-ft model
(tutorial MLP head, Poisson-multinomial, end-to-end, 30k steps) takes DNA plus the condition's pooled
RNA track (depth-scaled to 5e7 reads; log1p and box-smoothed features at 9/33/129/513 nt) and predicts
harringtonine (TIS) and CHX, both strands. Four folds hold out HeLa, K562, U2OS or RPE1 (all three
RPE1 conditions). Scored on val (chr7) with the concordance scorecard: Pearson r vs replicate 2 as a
percent of replicate 1 vs replicate 2, over MANE transcripts of expressed genes
(`data/p6/scorecard/heldout_val/summary.{tsv,md}`; ledger `.superpowers/sdd/2026-10-19-p6-rnacond/`).

Mean percent of replicate agreement over the six conditions (held-out model = the fold that never
saw that cell line; per-line models = experiment 1, trained on that line from DNA only):

| track | scale | held-out RNA-conditioned | seen | per-line base | per-line + CHX head | RNA scaled |
|---|---|---|---|---|---|---|
| TIS | gene | 98% | 98% | 89% | 89% | 96% |
| TIS | 30-nt bins | 71% | 72% | 68% | 69% | 12% |
| TIS | codon | 45% | 50% | 46% | 48% | 4% |
| TIS | 1 nt | 37% | 43% | 39% | 42% | 2% |
| CHX | gene | 98% | 98% | 89% | 89% | 97% |
| CHX | 30-nt bins | 84% | 86% | 80% | 82% | 38% |
| CHX | codon | 71% | 74% | 67% | 74% | 19% |
| CHX | 1 nt | 63% | 67% | 59% | 79% | 12% |
| CHX | reading frame | 59% | 60% | 62% | 82% | 54% |

- **Generalises to unseen cell lines.** Held-out scores are within 1-6 points of the folds that
  trained on the line, and at or above the per-line DNA-only models at gene and 30-nt scale.
- **Gene level is solved by the RNA input** (98% of replicate; the DNA-only models reach 89%), and the
  model adds a little over the RNA-scaled baseline (96-97%).
- **Shape along transcripts comes from the model, not the RNA** (30-nt: 71-84% vs 12-38% for scaled RNA).
- **Fine scale is the gap**: TIS codon/1-nt 37-45%, the start-peak problem of §12. Codon
  periodicity needs the CHX head of §12 (reading frame 82% vs 59%), which is not yet in this model.
- Caveats: the RNA input is matched RNA-input from the same lysate; val chromosome only; test (chr8/9)
  not yet scored.

## 14. P6 experiment 3 — loss/structure search and final test (2026-10-20 to 10-22)

Two-speed search (spec `docs/superpowers/specs/2026-10-20-p6-fast-loss-search-design.md`): a fast loop on a
transcript-space cache of frozen fine-tuned embeddings (~4 min/config; 156 configs in two phases,
`autoresearch/p6fast/ledger.md`) and full end-to-end confirmations on the leave-one-cell-line-out folds
(`.superpowers/sdd/2026-10-19-p6-rnacond/task-3-ledger.md`, `.superpowers/sdd/2026-10-20-p6-fastloop/progress.md`).

**Final recipe** (`--head rnacond_fc2d --local-weight 100 --local-libraries tis,chx --local-block 1024
--local2-block 32 --local2-libraries tis --local2-weight 300`), DNA + RNA input, harringtonine (TIS) + CHX out:
- factorised outputs for both tracks (amount per 1 kb block x softmax over positions in the block);
- local (1 kb) multinomial on both tracks — ribosome occupancy is compositional within a transcript region:
  starts compete for scanning ribosomes and elongating ribosomes share the local ORF;
- codon-scale (32-nt) multinomial on TIS — alternative starts sit within a few codons of each other;
- two stacked 9-nt convolutions over the embedding (start-codon and P-site frame context), plus a TIS-only
  dilated branch (~1 kb) for 5'UTR / uORF context that decides near-cognate use.
No hand-coded codon features, no annotation or Ribo-TISH calls in training.
Did not help: explicit 5'->3' scanning (hazard), TIS-CHX coupling, overdispersed noise, CHX frame loss,
deep/attention trunks without the local losses, extra RNA features.

**Test chromosomes (chr8/9, scored once), each cell line held out, percent of replicate 1 vs 2**
(final / reference RNA-conditioned model / RNA scaled):

| held-out line | TIS gene | TIS 30-nt | TIS codon | TIS 1 nt | CHX 30-nt | CHX codon | CHX 1 nt | CHX frame |
|---|---|---|---|---|---|---|---|---|
| K562 | 95 / 97 / 96 | 84 / 64 / 12 | **76** / 40 / 4 | **74** / 32 / 2 | 80 / 71 / 35 | 65 / 46 / 13 | 58 / 34 / 7 | 72 / 52 / 43 |
| U2OS | 95 / 96 / 93 | 87 / 67 / 10 | **77** / 44 / 4 | **74** / 33 / 2 | 82 / 74 / 32 | 70 / 51 / 14 | 66 / 40 / 8 | 76 / 52 / 47 |
| HeLa | 93 / 94 / 93 | 78 / 63 / 14 | **65** / 42 / 4 | **55** / 39 / 2 | 68 / 66 / 38 | 58 / 50 / 14 | 54 / 49 / 10 | 65 / 59 / 56 |
| RPE1 (3 cond.) | 98 / 99 / 98 | 91 / 75 / 11 | **93** / 49 / 4 | **97** / 40 / 2 | 101 / 95 / 40 | 116 / 92 / 22 | 130 / 88 / 16 | 88 / 63 / 58 |

Test agrees with validation within ~2 points everywhere (val K562 TIS codon/nt 74/73). RPE1 fine-scale CHX
exceeds 100% because its replicates agree poorly at that scale. Gene level costs 1-2 points vs the reference.

**Start sites from predicted tracks** (Ribo-TISH on predicted vs real K562 tracks, same synthetic pipeline,
depth-matched, 2 draws; K562 held out): test Jaccard **0.24-0.25** (precision 0.53, recall 0.31; ATG 0.35,
near-cognate 0.15; annotated starts recall 0.52-0.53, precision 0.93); same-pipeline ceiling (real data
resampled) 0.69. Val 0.23-0.25. Experiment 1 (per-line model trained on K562) was 0.15 (precision 0.27).
Peak at the exact start (val, K562): ATG 0.71 (real 0.68), near-cognate 0.20 (real 0.35).

**Where it stands vs the north star** (RNA-seq of an unseen line in -> base-resolution Ribo-seq out, good enough
to read starts): gene level ~95-98% and 30-nt shape 78-91% of replicate; TIS codon/nt 55-97%; AUG start
positions at real-data level; the remaining gap is near-cognate starts (about half of real) and start recall
from Ribo-TISH on predicted tracks (0.25 vs ~0.47 replicate agreement).

## Takeaways for downstream modeling

1. **Judge embeddings against the one-hot sequence floor (§4), not chance.** The honest
   foundation-model lift is 0.75→0.90 AUROC / 0.73→0.82 win@64bp. One-hot codon ≈ chance
   confirms the signal is contextual.
2. **Concatenate AlphaGenome + Evo2** — complementary at every scale; Evo2 carries the base
   resolution (the win@64 edge), AlphaGenome the regional context.
3. **Report the near-neighbour win-rate, not the global AUROC, as the headline** — it
   reflects actually calling a start codon and isn't inflated by regional priors.
4. **A linear head is a strong baseline — and stayed best under search _and_ at dense scale.**
   The autoresearch fleet (§6) found logistic best across all four metrics; §9 then confirms it
   against gradient-boosted trees at the true 2M imbalance — XGBoost and LightGBM both land within
   noise of logistic. This points toward frozen-representation experiments, but the single-seed
   result is not evidence of an absolute feature ceiling.
5. **The bottleneck is the representation, not the classifier.** No head beats logistic on the
   frozen `ag7` stack (§9), and the clean rerun holds at ≈0.30 recall (§10). Training a sparse
   binary head on frozen embeddings is the limiting design; the next phase fine-tunes the
   sequence model on raw Ribo-seq tracks and derives calls on top (see ROADMAP P6). The frozen
   `ag7` head is the baseline that model must beat.
6. **Near-cognate starts are the open problem.** Recall is 0.58 on AUG vs 0.07 on near-cognate
   (§10) — the novel biology is exactly where the current caller fails.

## Caveats

- **Protocol audit (2026-09-09):** historical recall-at-FP values optimized their threshold on
  TEST and are oracle ranking diagnostics, not deployable operating points. The current code
  selects a threshold on a disjoint validation subset. A five-fold transcript cross-fit over the
  saved TEST predictions reproduced the main oracle point for `Dense(None)` (0.2999 recall,
  0.821 FP/tx), but the clean calibration/operating/test protocol still requires a full rerun.
- The curated manifest contains 1,310 duplicated `(transcript_id, mrna_index)` positive rows.
  Current head paths retain one row per site; historical curated metrics implicitly up-weighted
  those positives. Historical efficiency regression also treated unexpressed/unmeasured rows as
  zero; the corrected target now excludes them.
- Single seed, single 60k train subsample, single chromosome split — solid for
  "does it work", but run proper cross-validation before any figure/claim.
- Negatives capped at 3× positives; the realistic genome-wide imbalance is larger —
  report metrics at the true imbalance for a calibrated claim.
- The store is per-candidate at offset 0 (AlphaGenome) and offsets {0,3,6,9} (Evo2);
  the Evo2 downstream-offset stack and AlphaGenome 131k were not yet swept for the
  dTIS stratum, where they may help most.

## Reproduce

```bash
eval "$(conda shell.bash hook)" && conda activate tisiago
python -m tisiago.eval        --store data/store        # tables 1 + 2
python -m tisiago.resolution  --store data/store        # table 3
```
