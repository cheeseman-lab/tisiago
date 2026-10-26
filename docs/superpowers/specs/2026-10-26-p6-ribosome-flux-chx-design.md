# P6 — ribosome-flux CHX output (design)

Status: chosen by Matteo 2026-10-26 ("option 2": attack CHX periodicity model-side).

## Why

5'UTR diagnosis (`.superpowers/sdd/2026-10-24-p6-utr5-diagnosis/progress.md`): at reproducible 5'UTR starts the
predicted TIS track is ~0.9 of replicate, but Ribo-TISH's CHX frame test fails on short uORFs (median 23 aa): uORFs
with in-frame CHX share > 0.5 are 0.18 predicted vs 0.71 real; global in-frame share 0.44 vs 0.62. Wider CHX context
(fast loop .15 -> .29, step 2 .18 -> .20) and a frame loss do not fix it. Per-position heads must re-derive the reading
frame everywhere; for a 23-codon ORF there is too little local evidence.

## Idea (biology)

Elongating ribosomes are in the frame of the start codon that loaded them, move one codon at a time, and leave at a
stop. So CHX density = in-frame propagation of initiation flux until termination. Make that the output structure:

    for each frame f, codon index k along the transcript (5'->3'):
        flux[k] = flux[k-1] * (1 - stop[k]) * keep + init[k] + bg
    init  = the model's own predicted TIS rate at that codon's first nt (option: stop_gradient)
    stop  = sigmoid(learned logit from the CHX features) — termination/drop-off hazard learned from sequence and
            context; NO hand-coded stop codons (the network has the one-hot branch and can learn them)
    keep  = learned global processivity in (0, 1]; bg = learned small background

Implemented as a linear recurrence over codons per frame (pad L to a multiple of 3, reshape (L/3, 3), associative
scan of (a, b) pairs, `jax.lax.associative_scan`), then a learned short frame kernel (P-site offset blur, 3-5 taps)
and a learned per-position dwell factor `exp(d[i])`. The flux enters the CHX output as:
- `mode: logit` — add `beta * log(flux_dwell + eps)` to the existing factorised CHX within-block logits (beta learned,
  init 0, so training starts from the current model);
- `mode: replace` — within-block CHX profile = normalised flux_dwell; block amounts unchanged.

The TIS -> CHX path couples initiation and elongation structurally (unless stop_gradient): the CHX loss also trains
the TIS head toward starts that explain downstream in-frame density.

## Evaluation

Fast loop first (frame now comes from TIS positions, not only from the frozen embeddings): cache
`data/p6/txcache/fc2d_k562`, base = `autoresearch/p6utr5/configs/u0_base.yaml` (step-1 head). Metrics:
`K562_utr5_orf_framed` (real .706, base .152), `K562_chx_frame` (base .700), CHX codon/nt, guard: primary3 and TIS
codon/nt within 1 point of base (3-seed means). Winner -> step 2 (`txtrain --head-config`), 3 seeds, Ribo-TISH 5'UTR
recall and Jaccard vs step 2 (0.17 / 0.367). Validation only; test once at the end.

## Caveat

Crops: flux from upstream of a crop is missing; transcripts longer than the fast loop's max length lose it at the
crop start (learned bg absorbs part). Report 5'UTR metrics, which sit near the cap, as the primary readout.
