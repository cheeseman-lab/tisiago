# P6 fast loss/structure search (design)

Status: approved by Matteo 2026-10-20 ("Let's do this!"). Follows FINDINGS §13 and the Task 3 TIS search
(`.superpowers/sdd/2026-10-19-p6-rnacond/task-3-ledger.md`: factorised TIS output + local multinomial loss lifts
K562-held-out TIS codon 41 -> 68%, nt 33 -> 66% of replicate).

## Goal

Find, quickly and in a principled way, the loss functions and input structure that let a model take RNA (plus
AlphaGenome sequence context) and predict base-resolution harringtonine (initiation) and CHX (elongation) tracks.
Full end-to-end fine-tunes take 3-4 h; the search must run in minutes per experiment, unattended for 3-4 days,
then the winning recipe is scaled to all folds and scored once on test.

## Two speeds

**Fast loop (minutes).** A fine-tuned backbone (`data/p6/runs/ar_rnacond_k562_v2/best`, trained with K562 held
out) is run once; its 1-bp embeddings are cached, PCA-reduced to 256 dims (fit on training transcripts), in
**transcript coordinates**: one record per MANE transcript of a gene expressed in a condition, cap to poly-A,
spliced, 5'->3'. Each record: embedding [L, 256] fp16, per condition the depth-scaled RNA [L], TIS and CHX
targets [L] for rep1, rep2 and pooled, plus metadata (transcript id, gene, chrom, split, CDS start/end in
transcript coordinates — CDS is for metrics only, never an input). Embeddings are condition-independent (the
backbone sees DNA only); RNA and targets are per condition. Train/val split as the main protocol (val = chr7;
test chr8/9 never cached). A head + loss trains on the cache in ~5-10 min on one GPU.

**Slow loop (hours).** Top fast-loop configurations are ported to the full rnacond model and fine-tuned end to
end on the K562-held-out fold (existing launcher), to check that the gain survives when the backbone learns
under the new loss. Promotion: a fast-loop config that beats the current fast-loop best by >= 3 points on the
primary score (mean of K562 TIS codon and nt percent) with guards intact.

## Fast-loop scoring (in memory, val transcripts)

Per condition and library, on transcript vectors exactly like `scorecard.py` (gene log1p CDS total; 30-nt bins;
codon bins in CDS frame; 1 nt; within-transcript scales = median over transcripts with >= 50 rep2 reads), as a
percent of rep1 vs rep2; plus TIS peak share at real Ribo-TISH starts (val only, for evaluation). Primary: K562
(held out) TIS codon + nt. Guards: K562 TIS/CHX gene and bin30 within 1 point of the fast-loop reference
(the reproduced rnacond head). Also reported for seen conditions. Sanity: the reproduced reference head on the
cache must land within a few points of the full-model scorecard (else the cache is wrong).

## Search space (organised by biological hypothesis)

1. Competition among starts: local / whole-transcript normalisation of TIS, block size, factorised
   amount x position output, temperature.
2. Scanning: initiation as a 5'->3' process along the mRNA (e.g. hazard/survival parameterisation:
   P(start at i) = h_i * prod_{j<i}(1 - h_j), learned hazard from embedding + RNA), no codon rules.
3. Initiation-elongation coupling: consistency between predicted starts and the downstream CHX rise.
4. Noise model: Poisson vs negative binomial / overdispersion, coverage weighting, rep1/rep2 as two draws.
5. Input structure: RNA features (raw, smoothed, one-sided, 5'-distance), receptive field and architecture of
   the head (MLP, dilated conv, attention along the transcript), CHX periodicity pieces (frame loss).
Constraint throughout: no hand-coded discrete sequence features (no start-codon/Kozak indicators), no Ribo-TISH
calls or labels in training.

## Infrastructure

- `tisiago.seq2func.txcache` (+ GPU script): build the cache. `tisiago.seq2func.fastloop`: config -> train ->
  score -> one results row; heads/losses as registered, composable components. JAX in `.venv/agft`.
- `autoresearch/p6fast/`: `program.md` (rules for the driver), `results.tsv`, `ledger.md`, configs; a Slurm
  launcher for one or a batch of configs per GPU job.
- Driver: an autonomous agent proposes configs by hypothesis, runs them on all GPU lanes within per-partition
  caps, records everything, promotes winners to slow runs, and stops after 3-4 days or on convergence.
