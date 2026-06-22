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

**Evo2 lift — matched `ag` vs `ag7` pair (2026-06-21).** Adding the Evo2 blk28 off{0,3,6,9}
stack to the AG+Kozak features, both trained at a **matched 1M-negative** cap (24:1; the big
node was occupied so 2M was not schedulable — §7's 2M AG above is the reference), best head
`Dense(None)`, same TEST @ 230:1:

| metric (TEST @ 230:1) | `ag` (1M) | **`ag7` (1M, +Evo2)** |
|---|---|---|
| AUPRC (base 0.0043) | 0.2424 | **0.2959** |
| recall @ ≤1 FP/tx | 0.247 | **0.309** |
| recall @ ≤5 FP/tx | 0.419 | **0.500** |
| recall @ ≤20 FP/tx | 0.692 | **0.753** |
| non-cognate mean p (→0) | 0.0004 | 0.0004 |
| cognate Brier | 0.0036 | 0.0035 |

The two levers do **different jobs**: imbalance-matched training already drove non-cognate to
≈0 (grounding, solved by AG alone), so Evo2's ~25% recall / +22% AUPRC lift is **cognate
discrimination** — its base resolution helps rank a true start above near-cognate decoys. Best
caller to date: **`ag7` Dense(None), recall 0.309 @ ≤1FP/tx at true 230:1, AUPRC 0.296 (69×
base), non-cognate ≈0.0004.** (Even the 3:1-trained `curated-C` improves with Evo2 — recall
0.036→0.142 — but stays far below imbalance-matched training.)

**Caveats (do not overclaim).** Single seed; the headline AG table is at 2M negatives while
the Evo2-lift pair is at a matched 1M (node-availability forced, not chosen); and the cleanest
baseline — *balanced-train + recalibrate-to-true-prior* — is not yet run. But the AUPRC/recall
gains are rank-based, so recalibration alone could not close most of the gap. This is strong
**preliminary** evidence for imbalance-matched training **and** a real Evo2 lift on cognate
discrimination; confirm across seeds and against the recalibrate baseline.

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

## Takeaways for downstream modeling

1. **Judge embeddings against the one-hot sequence floor (§4), not chance.** The honest
   foundation-model lift is 0.75→0.90 AUROC / 0.73→0.82 win@64bp. One-hot codon ≈ chance
   confirms the signal is contextual.
2. **Concatenate AlphaGenome + Evo2** — complementary at every scale; Evo2 carries the base
   resolution (the win@64 edge), AlphaGenome the regional context.
3. **Report the near-neighbour win-rate, not the global AUROC, as the headline** — it
   reflects actually calling a start codon and isn't inflated by regional priors.
4. **A linear head is a strong baseline — and stayed best under search.** The autoresearch
   fleet (§6) swept feature subsets × head × class-weighting against all four metrics; the
   winners are all logistic on a 19.6k-dim AG+Evo2+Kozak stack, ~0.02–0.05 over the 2-key
   baseline on every metric. Next: confirm the winners across seeds/splits (Phase 3), then the
   hard stratum (dTIS) and the per-condition efficiency labels (`max_norm_*`).

## Caveats

- Single seed, single 60k train subsample, single chromosome split — solid for
  "does it work", but run proper cross-validation before any figure/claim.
- Negatives capped at 3× positives; the realistic genome-wide imbalance is larger —
  report metrics at the true imbalance for a calibrated claim.
- The store is per-candidate at offset 0 (AlphaGenome) and offsets {0,3,6,9} (Evo2);
  the Evo2 downstream-offset stack and AlphaGenome 131k were not yet swept for the
  dTIS stratum, where they may help most.

## Reproduce

```bash
conda activate tisiago
python -m tisiago.eval        --store data/store        # tables 1 + 2
python -m tisiago.resolution  --store data/store        # table 3
```
