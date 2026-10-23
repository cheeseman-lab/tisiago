# P6 step 2 — backbone fine-tuned on spliced-transcript input (design)

Status: agreed direction 2026-10-23 ("1 then 2"); this spec fixes the design. Step 1 (genomic backbone + transcript-space
head on all folds) is the comparison point.

## Why

The near-cognate diagnosis (`.superpowers/sdd/2026-10-22-p6-nc-diagnosis/`) found the residual start-site misses are
spliced 5'->3' context: reproducible near-cognate starts in later exons are hit ~0.25-0.31 vs ~0.53-0.60 in the first
exon, and 5'UTR Ribo-TISH recall is 0.10-0.15. The deployed backbone sees genomic DNA (introns in between), so what a
scanning ribosome has passed (upstream exons, uORFs, the cap distance) is far away or interrupted. A transcript-space head
on frozen genomic embeddings recovers part of this (+8.7 primary3); the remaining step is to let the backbone itself see
the mature mRNA.

## Model

- Input: the spliced, sense-oriented mRNA sequence of one MANE transcript (cap -> poly-A), one-hot, padded with N to a
  length bucket (4096, 8192, 16384, 32768; longer transcripts are cropped to 32768 in training, tiled at inference with
  the same nearest-centre rule as `predict_tracks.owned_spans`). One transcript per example: no packing, so no context
  leaks between unrelated transcripts.
- Backbone: AlphaGenome via alphagenome-ft, trainable, initialised from the fold's genomic fine-tuned backbone
  (`data/p6/runs/ar_rnacond_<g>_fc2d_l2w300/best`); arm `rnacond:<G>` as before (same held-out conditions).
- Head: the step-1 transcript-space head (`p3r02_onehot41_l3`: conv trunk 9x2 + CHX trunk 25x2, factorised TIS/CHX 1 kb
  two grids, TIS dilated context 1-256, one-hot +-20 nt branch) on the backbone's 1-bp embeddings (1536-d, a learned
  linear projection to 256 in place of the cache PCA). Single strand: outputs sense TIS and CHX only.
- RNA input: the cache's depth-scaled pooled RNA 5' ends in transcript coordinates (sense, antisense), as the fast loop.
- Losses: exactly the step-1 head losses (sample Poisson; local multinomial 1 kb on TIS+CHX x300; global multinomial x5;
  local multinomial 32 nt and 3 nt on TIS x300). No new loss in this step: the only variable is the backbone's input.

## Data

Reuse the txcache: `index.parquet` (units, splits, junctions), shard `seq.npy`, `rna.npy`, `tis.npy`, `chx.npy`. No new
genome reads. Train = non-chr7/8/9 transcripts, val = chr7, test chr8/9 only via the step-1 final-test path (test cache,
`--i-am-final`). Expression masks as the cache (a transcript counts for a condition only where expressed).

## Evaluation (unchanged scorer)

Val predictions in the cache's transcript layout scored by the fast-loop scorer (`score_predictions`, fixed): K562 / each
held-out line TIS and CHX percent of replicate at gene / 30 nt / codon / nt + CHX frame, primary3, reproducible-start
NC/ATG hit ratios split first vs later exons. Export to genomic bigWigs (`export_bigwigs`) -> Ribo-TISH via
`predict_tracks --source float-resample` + `predicted_calls`: Jaccard / recall / precision overall, ATG, NC, annotated,
5'UTR.

Success = on the same folds and val transcripts as step 1: later-exon NC hit ratio and 5'UTR recall up, with TIS
codon/nt and CHX not more than 2 points below step 1. One fold (K562) first; all four folds only if K562 passes; test
once at the end with whichever architecture wins on val.

## Implementation units

1. `seq2func/txtrain.py`: transcript example iterator over a TxCache (bucketed crops, seq/rna/tis/chx/mask), the
   backbone + transcript head model (head reuses `fastloop` modules; embedding input dim configurable), train loop
   (AdamW, backbone lr 1e-4, head lr 1e-3, warmup-cosine, clip 1.0, 30k steps, eval on val every 1000 steps by the
   fast-loop scorer's primary3, keep best), checkpoint save/load, val prediction -> fastloop record + export.
2. `scripts/run_p6_txtrain.sh`: Slurm wrapper (GPU, agft env).
3. Tests on synthetic caches: bucket/crop/tiling round trip; strand/coordinate identity of exported tracks; a one-step
   smoke of the model with a tiny backbone stub; resume from checkpoint.

## Risks

- AlphaGenome was pretrained on genomic DNA; spliced input is off-distribution. Warm-starting from the genomic
  fine-tune and a low backbone lr mitigate; if K562 val falls below step 1 on TIS nt by >5 after 30k steps, stop.
- Memory at 32768 nt with a trainable backbone: batch 1 on A100 80 GB; A6000/L40S use buckets <= 16384 (longer
  transcripts cropped).
- MANE only (as step 1); other isoforms are a follow-up.
