# P6 — Seq2func on raw Ribo-seq tracks, TIS calls via Ribo-TISH

_Design spec, 2026-10-05. Status: approved for planning; implementation paused after the plan._

## 1. Why

The frozen line (P1–P5) trained a binary head on frozen AlphaGenome + Evo2 embeddings against
sparse, union-across-lines TIS labels. Under the clean protocol it reaches recall 0.306 at 0.99
FP/transcript (FINDINGS §10) and richer heads do not help (§9). It is mostly an AUG caller:
recall 0.58 on AUG vs **0.07 on near-cognate** starts. It also cannot say anything per cell line.

Following Peter Koo's advice, P6 fine-tunes AlphaGenome to predict the **raw Ribo-seq tracks**
(dense, quantitative supervision) and derives discrete calls on top. Calls come from **the same
caller that produced the labels** — Ribo-TISH, run the way Jimmy ran it — so a difference between
predicted and observed calls is attributable to the tracks, not to a different calling method.

## 2. Goals and success criteria

1. Predict stranded, base-resolution TIS (harringtonine), CHX and RNA-input tracks for each of
   6 conditions from sequence.
2. Derive per-condition TIS calls by running Jimmy's Ribo-TISH procedure on predicted tracks.
3. **Success** (held-out chr8/chr9, scored once): at the FP/transcript operating point the
   predicted-track calls achieve, beat the frozen `ag7` calls on **near-cognate recall** without
   losing **AUG recall** (frozen baseline scored at the same FP/transcript, union labels).
4. Settle the cell-line question with evidence: a shared multi-condition model vs per-condition
   models, decided by a rule fixed in §7 before any test-chromosome scoring.

## 3. Data (verified 2026-10-05)

Source: `/lab/barcheese01/aTIS_data/` (docs: `DATA.md`, `PROCESSING.md`; sample sheet
`ribosome_profiling/manifest.csv`).

| Item | Detail |
|---|---|
| Conditions | HeLa, K562, U2OS, RPE1_Async, RPE1_Que, RPE1_Sen (6) |
| Replicates | 2 per condition |
| Libraries per replicate | TIS RPF (harringtonine), CHX RPF, TIS input, CHX input (RNA) |
| BAMs | 48, STAR-aligned, sorted, indexed: `ribosome_profiling/bam/{ID}_Aligned.sortedByCoord.out.bam` |
| P-site offsets | Ribo-TISH `.para.py` per RPF BAM (e.g. `{32: 13, 33: 13, 34: 13}`; RPE1 hand-edited) |
| Reference | GRCh38 + GENCODE v49 (`reference/`), same build as the tisiago manifest |
| Existing calls | `ribotish/combined/{cond}_TIS_predict_all.txt` (pooled), `ribotish/per_rep/` |
| Excluded | iPSC (Chen et al. 2020): bedGraph only, one replicate, no Ribo-TISH run — optional external check only |

The bedGraphs are unstranded and of undocumented anchor (5′ end vs P-site); they are not used.
All tracks are rebuilt from BAMs.

**Jimmy's caller** (reference procedure, `scripts/processing.sh`):

```
ribotish quality -b <bam> -g <gtf> --th 0.60 -l 20,38 [-t]
ribotish predict -t <TIS BAMs> -b <CHX BAMs> --tispara <TIS paras> --ribopara <CHX paras>
                 -g <gtf> -f <fasta> --minaalen 3 --alt --seq --aaseq
filter: TISPvalue <= 0.01, RiboPvalue <= 0.01, FisherQvalue <= 0.05
swissisoform: NormTISCounts = TISCounts / gene RNA counts * 1e6 >= 0.1; MANE or TSL 1-3 transcripts
```

## 4. Phases

Each phase is one autonomous pass with a "done when" and returns for review.

**P6.0 — Tracks and round-trip contract (CPU).**
- Build stranded 1-bp bigWigs per condition × replicate × library: RPF reads placed at their
  P-site using the `.para.py` offsets; input (RNA) reads at their 5′ end. Also a pooled-replicate
  track per condition × library. Record library depths.
- Per-condition **expressed-gene table** from input counts (threshold chosen to match the
  swissisoform `expressed_*` flags where they exist).
- **Round-trip contract:** observed tracks → synthetic BAMs (§6) → Jimmy's `ribotish predict` +
  filters → compare with `ribotish/combined/{cond}_TIS_predict_all.txt`.
- Rebuild per-condition labels (filtered calls per condition) from this code path; this replaces
  the untraceable manifest-building step.
- QC: replicate concordance per track (the accuracy ceiling) and CHX 3-nt periodicity.
- *Done when:* the round trip reproduces ≥ 0.95 of Jimmy's filtered calls per condition (Jaccard
  over `(transcript, start)`), and the QC table exists.

**P6.0b — Approach search (no code).** How Borzoi, AlphaGenome, Enformer and Koo-lab work handle
many cell types (shared multi-output heads, per-cell-type heads, expression conditioning) and what
they report on cross-cell-type transfer. Output: a short note in `docs/`. Informs the arms; does
not change the §7 rule.

**P6.1 — Environment and smoke test (GPU).**
- uv env `.venv/agft` with `alphagenome-ft` (≥ 0.1.12, which fixes the fold-split bug),
  `alphagenome_research`, JAX for CUDA. Load the local `all_folds` checkpoint
  (`$ALPHAGENOME_WEIGHTS_PATH`); no Kaggle download.
- Custom head on the 1-bp `StandardHead` template; overfit one 131 kb region.
- *Done when:* training loss on that region falls by ≥ 90% and peak GPU memory is recorded.

**P6.2 — Heads-only arms (GPU).** Backbone frozen (`create_optimizer(..., heads_only=True)`;
`freeze_except_head` alone does not block backbone updates).
- **Shared arm:** one head, 36 outputs (6 conditions × 3 libraries × 2 strands).
- **Per-condition arm:** 6 heads, 6 outputs each, trained independently on the same windows.
- *Done when:* both arms have val (chr7) track metrics and val calls (§6), with the replicate
  ceiling alongside.

**P6.3 — Capacity ladder (GPU, conditional).** Only if heads-only plateaus below the ceiling:
LoRA adapters, then partial unfreeze of the decoder. Same windows, splits and metrics.

**P6.4 — Calls, comparison and decision.** Apply §7 on val; then score the chosen arm once on
chr8/chr9 against the frozen calls (`data/calls/ag_w8k_test_calls.parquet`).

## 5. Model and training

- **Base:** AlphaGenome `all_folds` via `alphagenome-ft`, 1-bp decoder embeddings.
- **Targets:** per condition × library (TIS, CHX, RNA) × strand, replicate-pooled, 1-bp.
- **Loss:** Poisson-multinomial per track (multinomial over positions for shape + Poisson on the
  window total), as in Borzoi/BPNet.
- **Expression masking:** for TIS and CHX tracks, mask the loss over genes not expressed in that
  condition (blank Ribo-seq there is missing data, not zero initiation). RNA tracks are never
  masked, so predicted RNA is the expression signal at inference.
- **Windows:** 131 kb genomic windows tiled over genes expressed in any condition, training
  chromosomes only. 32 kb is a cost ablation.
- **Augmentation:** reverse complement with strand-channel swap; ±small shifts.
- **Splits:** train = all chromosomes except chr7/8/9; val = chr7; test = chr8, chr9 — identical
  to the frozen baseline. Caveat: `all_folds` pretraining saw these sequences (never Ribo-seq).
- **Hardware:** A100 80 GB (`nvidia-A100-20`, ≤ 2 GPUs per user). Heads-only fits at batch 1
  (blog: 14–27 GB); full unfreeze needs ≥ 76 GB.
- **Code location:** `src/tisiago/seq2func/` in this repo, env `.venv/agft`. Data prep, synthetic
  BAMs and call scoring are CPU code with unit tests; gruyerenome stays inference-only.

## 6. Calls from predicted tracks

1. Predict TIS and CHX tracks over val/test transcripts for each condition.
2. **Synthetic BAMs:** scale each predicted track so its total equals the real library's depth
   over the same regions; round to integer counts; emit reads of a fixed length `L` with the
   5′ end at `P-site − offset` on the correct strand; write a matching `.para.py` `{L: offset}`.
3. Run Jimmy's `ribotish predict` (same flags) with predicted TIS as treatment and predicted CHX
   as background; apply his filters and the swissisoform NormTISCounts filter, using the predicted
   RNA track summed over the gene for the normalization.
4. Score against the rebuilt per-condition labels (P6.0): recall, precision, FP/transcript, split
   by AUG vs near-cognate. Also rank by TISPvalue to draw a recall-vs-FP curve.
5. **Frozen comparison:** union the per-condition predicted calls; score the frozen calls at the
   same FP/transcript (a threshold on `p_tis` chosen on chr7) against union labels.

The round-trip contract (P6.0) validates steps 2–3 on observed tracks before any model exists.

## 7. Pre-registered decision rule (shared vs per-condition)

Computed on **val (chr7) only**:

- **M1:** union recall at the FP/transcript the arm's Ribo-TISH calls produce, near-cognate and
  AUG reported separately.
- **M2:** mean over conditions × libraries of per-gene log-count Pearson, divided by the replicate
  ceiling.

The arm that wins both goes forward. If they split, the arm with the higher near-cognate recall
in M1 goes forward, because that is the stated scientific gap. Test chromosomes are scored once,
after the decision.

## 8. Risks

- **Periodicity:** Ribo-TISH's RiboPvalue tests 3-nt periodicity in the CHX signal. If predicted
  CHX tracks lack frame periodicity, real starts fail the filter. Measured in P6.2 on chr7
  (predicted vs observed frame fraction); mitigation is a frame-aware loss term or relaxing only
  the RiboPvalue filter, reported as a deviation.
- **Depth scaling:** synthetic BAM depth drives Ribo-TISH p-values. Scaling to the real library
  depth keeps statistics comparable; the round trip checks it on observed data.
- **Label noise:** negatives are uncalled, not proven negatives (positive-unlabelled).
- **Pretraining overlap:** `all_folds` saw the test chromosomes' sequence (not Ribo-seq); the
  frozen baseline shares this caveat, so the comparison stays fair.
- **Compute:** heads-only is cheap; LoRA/unfreeze is gated on evidence (P6.3).

## 9. Out of scope

- In-silico perturbation of the trained model (follow-up once a model passes §2).
- Ribo-TISH `tisdiff` differential calling between conditions.
- iPSC and other external datasets beyond an optional sanity check.
- Retraining or replacing the frozen `ag7` baseline.
