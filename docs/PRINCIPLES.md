# P6 principles — fine-tuning AlphaGenome for Ribo-seq tracks and training the head

What P6 learned (2026-10-05 to 10-27) about turning a cell line's RNA-seq into base-resolution harringtonine (TIS) and
cycloheximide (CHX) Ribo-seq. The principles are about the **track model**. Start calling (Ribo-TISH today) is a
separate layer on top, and the last section says what to redo when that layer changes. Evidence: FINDINGS §11-16,
`autoresearch/p6fast/ledger.md` (phases 1-3), `.superpowers/sdd/2026-10-2*-p6-*/`.

## 1. Problem framing

- **Condition on RNA, predict Ribo-seq, hold out the cell line.** RNA-seq is cheap for a new line; high-quality
  Ribo-seq is not. Train on the other lines, test on a line the model never saw (leave-one-line-out). Gene-level
  Ribo-seq is then mostly RNA (93-98% of replicate); everything interesting is the within-transcript shape.
- **Per-line models without RNA fit their own tracks but do not transfer**, and their starts were worse than the
  frozen-embedding baseline (§12). The RNA input is what makes the model a cross-line predictor.
- **Measure in percent of replicate agreement** (rep2 vs model as a percent of rep2 vs rep1) at gene / 30 nt / codon
  / nt, plus CHX in-frame share and reproducible-start hit rates (`scorecard.py`, `fastloop.score_predictions`). KL or
  composite scores hide how far from concordance the model is. Lines whose replicates disagree (RPE1) exceed 100%.
- **Separate initiation from pausing in start metrics.** Harringtonine leaves ribosomes at strong elongation pauses;
  most in-CDS in-frame "near-cognate starts" are pauses (§17). Report start metrics in-CDS vs outside the CDS.
- **Peak sharpness is a per-library property** (K562 sharpest, HeLa flattest; §17). A held-out line gets the training
  lines' average sharpness; compare against a replicate at equal depth, never against the reads that defined the sites.
- **Count only reproducible starts as truth for start metrics.** Only ~20% of near-cognate and ~46% of 5'UTR
  Ribo-TISH calls reproduce between replicates; report model hit rate on reproducible starts relative to rep1 -> rep2.

## 2. Inputs

- **RNA: depth-scaled pooled 5'-end coverage** (scaled to a 5e7-read library) into the head, not the backbone. CPM made
  the signal too small; transcript-relative RNA, one-sided/antisense RNA, and 5'-end sharpness features did not help.
- **DNA/mRNA sequence into the backbone.** Base AlphaGenome has little local near-cognate start information (local NC
  AUROC 0.65, below a one-hot ±20 nt floor of 0.76); fine-tuning under the track loss creates it (0.81).
- **Transcript (spliced, cap -> poly-A) coordinates are the largest single structural gain.** A head in transcript
  space on genomic embeddings: +8.7 primary, Ribo-TISH 0.25 -> 0.34 (§14-15). Feeding the backbone the spliced mRNA:
  further near-cognate gains past the first exon and fewer false calls (§15). Bio: the scanning ribosome reads the
  mature mRNA; introns are not context for start choice.
- **A ±20 nt one-hot branch into the TIS head** helps a little (+1.3 TIS); exact codon/Kozak identity sharpens peaks
  across junctions. Exon-structure inputs (exon index, junction distance) add nothing once context is spliced.
- **Never feed calls, CDS annotation, or learned-from-labels codon features.** The single allowed hand-coded rule is
  the genetic code's stop codons inside the CHX flux output (§6), a fixed law of translation, not a data feature.

## 3. Backbone

- **Fine-tune it.** Peak sharpness and near-cognate information do not come out of frozen embeddings with any head or
  loss tried (fast-loop phase 1: TIS peak share stuck at 0.10-0.11 vs real 0.23); they appear when the backbone learns
  under the loss. Backbone lr 1e-4, head lr 1e-3, AdamW, warmup + cosine, clip 1.0, 30k steps (60k = 30k).
- **Warm-start each stage from the previous fold-specific checkpoint** (genomic fine-tune -> spliced-input fine-tune).
  Never initialise a fold from a checkpoint that saw its held-out line.
- **One transcript per input, length buckets (4-32 kb), no packing** for spliced input; unrelated transcripts in one
  window leak context through attention.

## 4. Outputs (structure encodes biology)

- **Factorise every track as amount x position**: a total per 1 kb block times a softmax within the block, on two
  offset grids. Initiating and elongating ribosomes are a pool redistributed within a region.
- **Separate trunks for initiation and elongation.** TIS is local (start codon, Kozak, ±8 nt; conv 9x2) plus 5'
  context over hundreds of nt (TIS-only dilated context, receptive field ~1 kb: +5.5). Upstream-only context works as
  well as bidirectional, consistent with scanning. CHX needs its own wider trunk (conv 25x2: +2.4 nt, +.04 frame).
- **Make CHX frame a consequence of initiation, not a per-position guess** (§16): CHX within each block = the
  model's own predicted initiation propagated codon by codon in its frame and terminated at in-frame stop codons,
  times a learned dwell factor (`model.chx_flux`, replace mode, no frame blur). Doubles uORF framing on every line at
  ~equal track scores. A learned stop hazard alone did not find stops; frame losses alone never move frame.

## 5. Losses

- **Local multinomial (1 kb) on both tracks is the core loss** (42 -> 66 primary on its own; weight 300 at
  convergence; TIS-only local losses break CHX). Bio: starts compete for scanning ribosomes; elongating ribosomes
  share the local ORF.
- **Add a 32-nt TIS multinomial** (+4.3): alternative starts sit within a few codons of each other; codon-scale
  competition puts the peak on the exact codon. A 3-nt term is neutral-to-slightly positive.
- **Keep a Poisson on sample totals and a weak global multinomial** for amounts; per-nt Poisson/NB likelihoods,
  tempered targets, frame KL, coupling losses, scanning-hazard outputs, and training on replicate-reproducible signal
  only (removes real low-count starts) all failed.
- **Mask unexpressed genes** (blank Ribo-seq there is missing data, not zero initiation).

## 6. Process

- **Two speeds.** A fast loop on a frozen transcript-space embedding cache (~4 min/config) screens losses and
  structure; only winners go to full fine-tunes. Components that need backbone learning (sharpness) cannot be judged in
  the fast loop; components that do not depend on embeddings (output structure, flux) can.
- **Decide on validation (chr7) with 3 seeds; differences under ~1.5 points are noise.** Score test (chr8/9) once per
  model, after decisions, with an explicit `--i-am-final` gate; never tune on test.
- **Diagnose before building.** Each big step came from a diagnosis on validation: near-cognate misses -> spliced
  context (§14-15); 5'UTR misses -> CHX frame on short ORFs, not TIS placement (§16).

## 7. The calling layer (what to redo when it changes)

The track model is caller-independent; the numbers above in percent of replicate do not depend on Ribo-TISH. What does:

- **Run the caller identically on predicted and real tracks** (same synthetic-BAM pipeline, depth-matched), and report
  the real-vs-real (replicate) reference with the same caller. Calls are a property of model + caller together.
- **Choose the call cutoff on validation** (real call count or a target precision) and apply it once on test. For a
  truly new line the cutoff must come from the training lines' validation calls (the held-out line has no Ribo-seq).
  Structured CHX changes how often a frame test passes (flux CHX made Ribo-TISH over-call 1.2-2.6x); rank by the
  initiation evidence (TIS p) rather than the combined TIS+frame score.
- **Report per start class** (ATG / near-cognate; annotated / 5'UTR / internal) with precision next to recall.
- **When the caller changes:** rerun only `predicted_calls` (and the replicate reference) on the saved exports
  (`data/p6/txtrain/export*/`), re-choose the cutoff on validation, score test once. Retrain the model only if a
  diagnosis shows the caller needs a track feature the model does not produce (as CHX frame on uORFs did).
- Known caller issue: Ribo-TISH's TIS-background grouping (`zbio/ribo.py`, `if s >= ctotal*parts[ip]`) leaves the
  last group `None` when one transcript holds >10% of TIS reads (U2OS/RPE1 chr7); fix with `while`.

## Current recipe (2026-10-27)

Fold-specific genomic fine-tune `rnacond_fc2d` (local 1 kb multinomial both tracks w100, 32-nt TIS w300) -> warm-start
`txtrain` on spliced mRNA with head `autoresearch/p6flux/configs/g7_replace_fixed_grad_k1.yaml` (conv trunks, TIS
dilated context to 256, ±20 nt one-hot TIS branch, factorised outputs, 1 kb / 32 / 3-nt local multinomials, flux CHX
with fixed stops). Open: TIS peak specificity (peaks ~0.7x as sharp as real; 5'UTR call precision ~0.25) and HeLa.
