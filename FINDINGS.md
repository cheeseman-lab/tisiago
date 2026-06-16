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

## 4. At true imbalance — the honest caller number (Phase 2, **preliminary, AG-only**)

Sections 1–3 are on the curated 3:1 decoy set. Phase 2 scores *every codon* in the
held-out transcripts (dense scan, GENCODE v49) and applies the calibrated head at the
**realistic** imbalance. First result is **AlphaGenome-only** (Evo2 extraction still
running — combined number pending):

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

**Grounding is only partial for AG alone.** Non-cognate codons (never trained on) are *not*
called (FPR 0.0016 at the operating threshold) but are *not* crushed to ≈0 either (mean p
0.12, p95 0.48) — AlphaGenome's 128 bp-upsampled embedding leaks regional probability onto
non-cognate positions. Evo2 (1 token/bp, true codon identity) is expected to lift both the
recall and the grounding; the combined AG+Evo2 number is the real Phase-2 headline.

## Takeaways for downstream modeling

1. **Concatenate AlphaGenome + Evo2** — complementary at every scale.
2. **Report the near-neighbour win-rate, not the global AUROC, as the headline metric** —
   it reflects actually calling a start codon and isn't inflated by regional priors.
3. **A linear head is enough** — spend effort on labels (per-condition translational
   efficiency from `max_norm_*`) and the hard stratum (dTIS), not on architecture.

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
