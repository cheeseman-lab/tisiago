# P6 fast search ledger

## Phase 2 (near-cognate start recall; primary2)
primary2 = mean(K562 TIS nt %, min(100, 100 x K562 near-cognate argmax-at-start / real)). Guards: phase-1
guards + K562 TIS codon >= 74.4 and nt >= 72.7 (phase-1 leader r20_l300_chx25x2 3-seed 76.4 / 74.7 minus 2).
Start class = codon at the real Ribo-TISH call (ATG vs any other = nc). GPU: nvidia-A100-20 only.
Stop: 3 consecutive rounds across >= 3 families without a primary2 gain > 2 (3-seed means), or 2026-10-24 18:00.

### Phase 2 leaderboard
(pending baseline)

### Phase 2 findings

### Phase 2 rounds
- 2026-10-21 09:40 | scorer | added per-start-class peak columns (`<cond>_tis_peak_{meta,argmax0}_{atg,nc}[_real]`) and `primary2`, appended after the phase-1 columns (old rows nan); unit-tested; commit 532256b. Phase-2 components (local count_power / target_power, model.tis_context, rna_onset / rna_relative inputs) commit ec94e45. |
- 2026-10-21 10:45 | P2R00 | baseline | phase-1 leader r20_l300_chx25x2 rescored, 3 seeds | configs/p2r00/* | 11174941 (time limit: seed 0 lost; submit.sh now 15 min/config), seed 0 rerun in 11175048 | s1 primary2 60.4, s2 60.9 (TIS codon 76.2 nt 74.6; K562 nc argmax0 0.145 vs real 0.311 = 47%; ATG argmax0 0.70 vs real 0.66; nc meta 0.058 vs 0.170, ATG meta 0.19 vs 0.33) | The gap is entirely near-cognate: ATG argmax is already at real level. |
- 2026-10-21 10:45 | P2R01 | fam 7 (sharpening, existing components) | per-nt Poisson on TIS (w10, w100); extra 32-nt local TIS multinomial w300; TIS temperature 0.5 | configs/p2r01/* | 11174942 (temp05 lost to time limit) | pnt10 60.5, pnt100 60.8 (no change); **l32tis300 62.9** (nc argmax0 0.161; FAIL tis_codon 74.2 < 74.4) | Per-nt Poisson does nothing for nc. A fine-scale (32 nt) TIS competition term gives +2.3 primary2 (single seed) at a small TIS codon cost -> R03 block/weight sweep + seeds. |
- 2026-10-21 11:55 | P2R02 | fam 7 (block weighting/tempering, TIS-only), fam 8 (TIS-only dilated context, RF ~1 kb), fam 9 (rna_onset, rna_relative) | configs/p2r02/* | 11175048 (+ leader s0), 11175056 | leader 3 seeds **60.6+-0.3** (60.5/60.4/60.9; TIS codon 76.4 nt 74.7); split control 61.7; tcp05 62.5; tcp0 60.7 FAIL tis; ttp05 57.2 FAIL tis (TIS nt 65); **tisdil 64.5** (nc argmax0 0.162, TIS codon 79.4 nt 77.0, phase-1 primary 78.2, CHX unchanged); onset 60.9; relative 57.9 (nc 0.132) | Long receptive field on the TIS branch is the first phase-2 lead (+3.9, also +2.8 phase-1 primary) -> R04 seeds + extensions. Block weighting is inside noise of its split control; tempering the target costs shape. RNA 5'-end sharpness / relative coverage add nothing (fam 9 negative). |

## Final summary (phase 1)
Stopped 2026-10-20 21:40 on the stop rule (b): rounds R18 (fams 1,2), R19 (fam 4) and R20 (fams 3,5,6)
gave no primary gain > 1.5 over the leader (3-seed means). 20 rounds, 118 configs, 14.2 GPU-hours.

**Leader (TIS):** `r14_lall300` / `r17_lall300_s1/_s2`, primary **75.2 +- 0.2**.
**Recommended recipe (same TIS, better CHX):** `r20_l300_chx25x2` (+ `_s1/_s2`), primary **75.6 +- 0.2**.
```yaml
train: {steps: 30000}
inputs: [embedding, rna, rna_onesided]      # rna_onesided dispensable (R10)
model:
  trunk: {kind: conv, hidden: 256, layers: 2, width: 9}
  chx_trunk: {kind: conv, hidden: 256, layers: 2, width: 25}
  chx_contrast: false
  tis: {kind: factorised, block: 1024, two_grids: true}
  chx: {kind: factorised, block: 1024, two_grids: true}
losses:
  - {kind: poisson, scope: sample, weight: 1.0}
  - {kind: multinomial, scope: local, block: 1024, libraries: [tis, chx], weight: 300}
  - {kind: multinomial, scope: global, weight: 5.0}
```

K562 held out, val chr7, percent of replicate (3-seed means):
| | primary | TIS gene | TIS bin30 | TIS codon | TIS nt | CHX gene | CHX bin30 | CHX codon | CHX nt | CHX frame | TIS peak meta / argmax0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recipe (lall300 + CHX trunk conv25x2) | **75.6+-0.2** | 97.7 | 86.5 | 76.4 | 74.7 | 98.8 | 82.2 | 66.9 | 61.1 | 0.68 | 0.11 / 0.36 |
| leader lall300 | 75.2+-0.2 | 98.2 | 86.8 | 75.3 | 75.1 | 98.7 | 82.3 | 66.2 | 58.8 | 0.64 | 0.11 / 0.35 |
| slow-leader fast equivalent (foc_lw20) | 66.8+-0.3 | 97.7 | 83.9 | 68.0 | 65.6 | 97.6 | 71.0 | 46.4 | 33.9 | 0.48 | 0.10 / 0.36 |
| seeded t3_foc_lw5_ps | 65.9+-1.0 | 97.5 | 83.0 | 66.8 | 64.4 | 97.6 | 70.3 | 46.1 | 33.6 | 0.48 | 0.10 / 0.36 |
| ref_rnacond_ps | 42.5+-0.5 | 97.5 | 69.7 | 45.7 | 39.3 | 97.7 | 73.4 | 49.9 | 38.3 | 0.51 | 0.05 / 0.30 |
(real K562 peak: meta 0.233, argmax0 0.446.) All guards pass with margin (CHX codon +17, nt +23 over ref).
Seen lines (leader): HeLa TIS codon/nt 62/53, U2OS 77/77, RPE1 90-103 / 92-115: K562 held out sits
between seen lines, so the limit is fit/information, not cross-line generalisation.

**Components that mattered (ablation path, 3-seed means):**
1. Local (1 kb) multinomial on **both** tracks (42.5 -> 66.2 at w20, 67.1 at w100; 75.2 at w300 with the rest).
   TIS-only local losses (the seeded leader and the slow leader) fail the CHX guards in every trunk.
   Bio: ribosome occupancy is a conserved pool redistributed within a transcript region: starts compete
   for scanning ribosomes, and elongating ribosomes share the local ORF, so both tracks are
   compositional within ~1 kb, not only in total.
2. Width-9 convolution over the embedding (+3.5: 67.1 -> 70.6), two stacked (71.7; RF 17 nt).
   Bio: the start codon, its Kozak-scale context (+-4-8 nt) and the P-site frame are read from
   neighbouring nucleotides, which a per-nt head cannot see.
3. Training to convergence: 30k steps (+2.3: 71.7 -> 73.9); 60k = 30k. (Optimisation, not biology.)
4. Factorised CHX output (amount x within-1 kb softmax): CHX bin30/codon/nt +3/+4/+3.5 at no TIS cost.
   Bio: elongating ribosomes, like initiating ones, are a total times a positional distribution.
5. Local weight 300 at convergence (+1.9 over 100). TIS saturates at 300; CHX keeps rising to 3000.
6. Separate wider CHX trunk (conv25x2): CHX nt +2.4, frame +.04, TIS unchanged. Bio: the elongation
   frame is a 3-nt periodicity integrated over tens of codons; initiation is local.

**What failed (no gain or worse, tested at 10k and again at 30k on the improved base):**
scanning hazard output (= block softmax); cap-anchored transcript-level TIS softmax (74.7, = block);
coupling to the predicted CHX rise (neutral at 0.5-1, -3 at 5); per-nt NB / Poisson likelihoods;
codon-phase (frame) KL on CHX or TIS (frame never moves without wider receptive field); multi-scale
local losses (32-256 nt); split trunk at equal config; deeper plain conv stacks (x3, x4) and wider
hidden (512); CHX-contrast input; one-sided / antisense RNA, position and whitened embedding inputs;
crop 8 kb, uniform sampling, lr 3e-3, batch 32. TIS peak meta share stays at 0.10-0.11 in every
config (real 0.233): sharpness at the exact start nt is not reachable from the frozen v2 PCA
embeddings with any head/loss tried — this is the job of the slow tier (backbone learns under the loss).

**Recommended recipe to scale:** promote `promote/r20_l300_chx25x2.md` to the slow tier (rnacond
fine-tune, K562 fold first): new head kind = factorised TIS + factorised CHX on Conv1D(256, 9) x2,
CHX on its own Conv1D(256, 25) x2 stack, no CHX contrast; `--local-weight 300` (also 100 as an arm,
since the slow tier's Poisson is on window totals) `--local-libraries tis,chx --local-block 1024`.
If the slow tier confirms, scale to all four folds and score once on test.

## Leaderboard
(guard-passing only; mean of seeds; K562 held out)
| name | primary | K562 TIS codon | nt | CHX bin30 / codon / nt / frame | guards | seeds |
|---|---|---|---|---|---|---|
| r20_l300_chx25x2 (recipe) | 75.6+-0.2 | 76.4 | 74.7 | 82.2 / 66.9 / 61.1 / .68 | pass | 3 |
| lall300 30k (r14_lall300, r17 s1/s2) (leader) | 75.2+-0.2 | 75.3 | 75.1 | 82.3 / 66.2 / 58.8 / .64 | pass | 3 |
| r17_lall3000 | 75.3 | 76.1 | 74.6 | 83.1 / 67.9 / 60.1 / .65 | pass | 1 |
| r17_lall1000 | 75.0 | 75.8 | 74.2 | 82.1 / 67.3 / 60.2 / .64 | pass | 1 |
| r19 nb1 (+lall300) | 74.7 | 75.4 | 74.0 | 82.7 / 65.9 / 59.1 / .64 | pass | 2 |
| txtis (R15/R18, cap-anchored TIS) | 74.7+-1.0 | 75.3 | 74.1 | 81.9 / 65.3 / 57.7 / .63 | pass | 3 |
| c9x2 30k softplus CHX | 73.9+-1.2 | 74.9 | 73.0 | 79.2 / 61.2 / 54.2 / .63 | pass | 3 |
| base 30k (factorised CHX, lall100) | 73.3+-1.0 | 73.8 | 72.7 | 82.0 / 65.2 / 57.7 / .63 | pass | 3 |
| conv9x2 10k | 71.7+-1.0 | 72.5 | 70.9 | 78.5 / 59.3 / 51.9 / .63 | pass | 3 |
| conv9 10k | 70.6+-0.3 | 71.7 | 69.4 | 78.2 / 57.7 / 47.6 / .60 | pass | 3 |
| lall100 10k | 67.1+-0.7 | 68.8 | 65.5 | 77.4 / 56.4 / 45.1 / .56 | pass | 3 |
| ref_rnacond_ps | 42.5+-0.5 | 45.7 | 39.3 | 73.4 / 49.9 / 38.3 / .51 | ref | 2 |
| (fail) foc_lw20 = slow-leader equivalent | 66.8+-0.3 | 68.0 | 65.6 | 71.0 / 46.4 / 33.9 / .48 | FAIL CHX | 3 |

## Findings
- Fam 1 (weight at convergence): local weight 300 > 100 > 30 at 30k steps (TIS saturates at 300; CHX rises to 3000). Transcript-level (cap-anchored) TIS competition ties the 1 kb block.
- Fam 6 (CHX periodicity): a separate wide CHX trunk (conv25x2) is the only thing that moves CHX frame (.64 -> .68) and nt (+2.4); frame losses never do. Bio: the elongation frame is integrated over tens of codons.
- Fams 2/3/4 at convergence: hazard, coupling, NB / per-nt Poisson remain neutral on the 75-point base.
- Fam 1 (output): a factorised (amount x within-1kb softmax) CHX output beats softplus CHX by +3-5 points at bin30/codon/nt. Bio: elongating ribosomes, like initiating ones, are a conserved pool redistributed along the transcript, so CHX is also best modelled as a total times a positional distribution.
- Fam 5 (inputs): one-sided RNA, antisense RNA and embedding whitening are each dispensable (within noise); the sense RNA + embedding carry it.
- Fam 1 (scale of competition): adding 32-256 nt local multinomials on top of the 1 kb block is neutral; 1 kb normalisation already captures the competition.
- Optimisation: 30k steps > 10k by 2-3 points (lr/batch neutral): fast-loop comparisons at 10k understate converged heads.
- Fam 5 (input structure): a per-nt MLP on the frozen embedding plateaus (~67, deeper/wider adds nothing); a width-9 conv over the embedding lifts TIS nt +4 and CHX frame 0.56->0.60. Bio: start recognition and the P-site frame are read from the codon and its immediate (Kozak-scale) neighbourhood, which a per-nt head cannot see.
- Fam 2 (scanning): hazard 5'->3' output ties the block-softmax factorised output (65.7-65.9 vs 66.2); start crops tie random crops. The competition is captured by any normalised parameterisation; explicit scanning adds nothing at this embedding.
- Fam 3 (coupling): KL of TIS to the predicted CHX rise is neutral at weight 1 and hurts at 5 (-3): predicted CHX rises are too smooth to locate starts.
- Fam 4 (noise): an added NB per-nt likelihood is neutral.
- Fam 6 (frame): codon-phase KL on CHX (10, 50) leaves frame agreement at 0.54: the frozen embeddings carry little reading-frame signal for the head to use.
- Fam 1: local (1 kb) multinomial on TIS gives the start-peak gain; its weight 5->50 adds only ~2 points in the fast loop. Adding the same local multinomial on CHX removes the CHX cost and lifts CHX above ref (bio: elongating ribosome density is shaped within a transcript region, so both tracks are compositional within ~1 kb, not just in total).
- Seed check: t3_foc_lw5_ps fails the K562 CHX guards (bin30 70.4 vs 73.6; codon 45.8 vs 50.1; nt 33.5 vs 38.7). Fixing the CHX cost of the factorised/local TIS head is a precondition for any winner.

## Rounds
- 2026-10-20 11:30 | R01 | fam 1 (competition) + noise floor | reproduce local weight 20; sweep 10/50; local on CHX too; plain softplus + local 20; seeds 1-2 of foc_lw5 and ref | configs/r01/* | jobs 11167834 (lw20, lw10, lw50, lall20), 11167835 (lw5_s1, lw5_s2, ref_s1, plain_lw20) | foc_lw5 3 seeds 65.9+-1.0 (64.9/65.0/66.9); lw10 66.3, lw20 66.7, lw50 67.8 (all FAIL CHX guards: bin30 ~71, codon ~46); lall20 64.9 PASS (CHX bin30 74.9, codon 52.1, nt 40.4 > ref); plain_lw20 53.5 pass; ref_s1 42.9 (ref 42.1) | lw20 reproduces only weakly in the fast loop (+0.8 over lw5, inside noise). Guarded leader = foc_lall20. Next: lall weights up, split CHX trunk. Added model.split_trunk (+tests). |
- 2026-10-20 11:50 | R02 | fam 1 (competition on both tracks) + fam 5 (split trunk) | does local CHX multinomial let TIS local weight rise without CHX cost; is the CHX cost from the shared trunk | configs/r02/* (lall50, lall100, t50c20, lall20_s1/s2, lw50_split, lall20_split, lall50_split) | 11167876, 11167877 | lall20 3 seeds 66.2+-1.3 pass; lall50 67.4 pass; lall100 67.9 pass (CHX bin30 77.8 codon 56.8 nt 45.4); t50c20 67.9 pass; lw50_split 68.2 FAIL chx_nt; lall20_split 67.3, lall50_split 67.4 pass | local-both scales with weight and keeps improving CHX; split trunk no TIS gain (CHX cost is a loss problem, not a capacity-sharing one). Leader lall100 (1 seed). |
- 2026-10-20 12:00 | R03 | fam 2 (hazard, crop start), fam 3 (coupling 1/5), fam 4 (nb), fam 6 (CHX frame 10/50) on lall20 base (66.2+-1.3) | configs/r03/* | 11167889, 11167890 | hazard_start 65.9, hazard_rand 65.7, lall20_start 67.2, coup1 65.4, coup5 63.2, nb1 65.8, frame10 65.1, frame50 65.7; all pass | NO GAIN (all within noise; coupling 5 hurts). Hazard = factorised; frame loss does not move CHX frame (0.54) -> frame info likely absent from the frozen PCA embeddings. No-gain streak 1 (fam 2,3,4,6). |
- 2026-10-20 12:20 | R04 | fam 1 (lall300, block 256 output/loss) + fam 5 (conv9, mlp 2x512, position) on lall100 base; lall100 seeds 1-2 | configs/r04/* | 11168033, 11168034 | lall300 67.8; b256 (output+loss) 66.1; lb256 (loss only) 66.6; lall100_s1 66.6; **conv9 70.5** (TIS codon 70.8 nt 70.1, CHX codon 57.7 nt 47.5, frame 0.60); mlp2x512 67.5; pos (pending); all pass | Weight saturates at 100. Local sequence context (conv, +-4 nt) is the first structural gain: +3 over lall100. Depth/width without context does nothing. -> R06 conv width/depth + seeds. |
- 2026-10-20 12:58 | R05 | fam 1 (multi-scale local competition: + blocks 128 / 32 / 256+64 on both, 64 on TIS) + optimisation (30k steps, lr 3e-3, batch 32) on lall100 base | configs/r05/* | 11170638, 11170639 (earlier ids cancelled unrun) | ms128 67.1, ms32 67.7, ms256_64 67.2, ms64tis 68.0 (base lall100 67.1+-0.7): no gain. steps30k 69.2, ms128_30k 70.0 (+2-3); lr3e-3 67.3; batch32 67.2; all pass | Multi-scale local competition adds nothing over the 1 kb block (the nt shape is limited by information, not by the loss scale). The head is under-trained at 10k steps: 30k gives +2-3. |
- 2026-10-20 13:12 | R06 | fam 5 (receptive field): conv width 3/9/21/51, conv9 x2 layers, conv9 hidden 512, conv9 + CHX frame 50, conv9 seeds 1-2 (lall100 base) | configs/r06/* | 11170636, 11170637 (any of A100/L40S/A6000; earlier ids cancelled unrun) | conv9 3 seeds 70.6+-0.3 (70.5/70.4/70.9); conv3 68.0; conv21 70.4 (CHX frame .64 nt 50.1); conv51 70.1 (CHX frame .67 nt 51.6, TIS peak argmax .33); **conv9x2 72.9**; conv9 h512 69.7; conv9+frame50 69.5; all pass | conv9 beats lall100 by +3.5 (3-seed). Width >9 helps CHX (frame) but not TIS; 2 stacked conv9 +2.3 more. Frame loss still useless. -> R08 conv depth. |
- 2026-10-20 15:09 | R07 | fam 5 (codon-aligned dilations 1/3/9/27 vs 1/2/4/8, conv15) + ablations on conv9 (no CHX contrast; local on TIS only) + seeds 1-2 of r01_foc_lw20 (fast-loop equivalent of the slow leader, for promotion) | configs/r07/* | 11170691, 11170692 | dilcodon 70.7, dil1248 71.0, conv15 69.5, conv9_nocontrast 70.6, conv9_tisonly 71.3 (FAIL chx_nt, CHX codon 47.6); foc_lw20 s1 66.6 (s0 66.7; FAIL CHX) | Codon-aligned dilations = ordinary ones = conv9 (no frame-specific gain). CHX contrast input is dispensable with a conv trunk. Local loss on CHX is needed for the CHX guards in every trunk. Slow-leader fast equivalent ~66.6. |
- 2026-10-20 15:19 | R08 | fam 5 (conv depth): conv9x2 seeds 1-2, conv9x3, conv9x4, conv5x3, conv15x2, conv9x2 hidden 128 / 512 | configs/r08/* | 11170968, 11170969 | conv9x2 3 seeds 71.7+-1.0 (72.9/71.4/70.9; CHX nt 51.9, frame .63); conv9x3 69.2; conv9x4 68.4; conv5x3 71.3; conv15x2 69.4; conv9x2 h128 71.6; h512 73.3; all pass | Two stacked conv9 (RF 17) is the depth optimum (+1.1 over conv9, inside noise); deeper plain conv stacks degrade. Hidden width irrelevant. TIS peak meta share stays 0.10 in every config. |
- 2026-10-20 16:27 | R09 | fam 5 + fam 6 (separate wide CHX trunk for elongation/frame) + optimisation on conv9x2 base: 30k steps; chx_trunk conv51; chx_trunk conv21x2; no CHX contrast | configs/r09/* | 11171706 | c9x2_30k **75.2** (TIS codon 76.5 nt 73.8; CHX nt 54.1); chx_trunk conv51 71.9 (CHX frame .65); chx_trunk conv21x2 71.6; nocontrast 72.1 (base conv9x2 71.7+-1.0); all pass | Training length is the lever (+3.5 at 30k). A separate wide CHX trunk does not lift CHX frame beyond .65. CHX contrast dispensable again -> drop it. |
- 2026-10-20 16:55 | R10 | fam 1 (factorised CHX output; TIS temperature 0.5; TIS block 2048) + fam 5 (crop 8 kb; uniform transcript sampling; whitened embedding; sense-only RNA; no one-sided RNA) on conv9x2 base | configs/r10/* | 11172205, 11172206 | chxfact 71.5 (CHX bin30 81.5 codon 64.2 nt 55.5: +3/+5/+4 over softplus CHX); temp0.5 71.7; tisb2048 72.5; crop8k 69.2; uniform 71.0; white 70.8; rnasense 70.7; noonesided 71.3 (base 71.7+-1.0); all pass | No TIS gain from any output/input variant (streak 1 since the R09 gain). Factorised CHX is a large CHX gain at no TIS cost -> adopt. One-sided RNA, antisense RNA, whitening, temperature: dispensable. |
- 2026-10-20 17:00 | R11 | fam 5 seeds (conv9x2 h512 s1-2) + fam 6 on TIS (codon-phase KL on harringtonine, weight 20/100) | configs/r11/* | 11172247 | h512 s1 70.7, s2 71.1 (s0 73.3; mean 71.7 = h256); tisframe20 71.5; tisframe100 72.0; all pass | No gain (streak 2: fams 1,5,6). Codon-phase KL on TIS does not sharpen peaks (meta .10-.11). |
- 2026-10-20 17:35 | R12 | optimisation + fam 1 (factorised CHX): conv9x2 + factorised CHX at 30k (3 seeds) and 60k; conv9x2 softplus-CHX 30k seeds 1-2 | configs/r12/* | 11172453, 11172454 | cf30k 3 seeds 73.4+-0.5 (74.0/73.0/73.3; CHX bin30 82.0 codon 65.3 nt 57.6); cf60k 74.2; c9x2_30k (softplus CHX) 3 seeds 73.9+-1.2 (75.2/74.0/72.7; CHX codon 61.2 nt 54.2) | 30k > 10k (+1.7 over conv9x2 3-seed); 60k = 30k (saturated). Factorised CHX keeps its CHX gain at 30k. |
- 2026-10-20 18:05 | R13 | fam 2 (hazard), fam 3 (coupling 0.5), fam 4 (NB) re-tested on the new base (conv9x2, factorised TIS+CHX, no contrast) at 30k steps | configs/r13/* | 11172561, 11172562 | base 74.3; hazard 74.0; nb1 75.1; coup05 73.4; all pass (ref point cf30k 3 seeds 73.4+-0.5, contrast on) | NO GAIN >1.5 (streak 1: fams 2,3,4). At convergence, scanning hazard = block softmax; NB term +1 (noise); coupling neutral/negative. |
- 2026-10-20 18:30 | R14 | fam 5 (hidden 512, conv9x3) + fam 1 (local weight 30 / 300) on the new base at 30k | configs/r14/* | 11172601, 11172602 | h512 73.5; c9x3 72.8; lall30 71.6; **lall300 75.1** (CHX codon 66.6 nt 59.3); all pass | Capacity/depth neutral at 30k. Local weight still monotone at convergence (30 < 100 < 300) -> R17 seeds + 1000/3000. |
- 2026-10-20 18:50 | R15 | fam 1/2 (competition scale: TIS softmax over the whole 5' crop with start crops, with and without the 1 kb local TIS loss -> crop-level TIS multinomial; TIS block 2048; crop-level instead of batch-level global multinomial) on the 30k base | configs/r15/* | 11172630, 11172631 | tiscrop_start (crop softmax + 1 kb local TIS loss) 73.9; **tisnolocal_crop (crop softmax + crop-level TIS multinomial, start crops) 75.8**; tisb2048 73.4; samplemn 74.6; all pass (base ~73.7) | Transcript-level (cap-anchored) TIS competition +2 over the 1 kb block, single seed -> R18 seeds. 1 kb block loss on TIS conflicts with a crop-wide softmax. |
- 2026-10-20 19:05 | R16 | seeds 1-2 of r13_base (the candidate recipe) + fam 6 (CHX frame 50 with factorised CHX) + fam 5 (position input) at 30k | configs/r16/* | 11172637, 11172638 | base 3 seeds 73.3+-1.0 (74.3/72.3/73.1; CHX bin30 82.0 codon 65.2 nt 57.7); chxframe50 73.9 (CHX frame .63); pos 73.3; all pass | NO GAIN from frame (fam 6) or position (fam 5) at convergence. Base recipe confirmed at 73.3. |
- 2026-10-20 19:15 | R17 | fam 1 (local weight at convergence): lall300 seeds 1-2, lall1000, lall3000 at 30k | configs/r17/* | 11172648, 11172649 | lall300 3 seeds **75.2+-0.2** (75.1/75.5/75.1; CHX bin30 82.3 codon 66.2 nt 58.7); lall1000 75.0 (CHX codon 67.3 nt 60.2); lall3000 75.3 (CHX bin30 83.1 codon 67.9 nt 60.1); all pass | GAIN +1.9 over the 73.3 base (3-seed) -> new leader lall300; streak reset. TIS saturates at 300; CHX keeps rising to 1000-3000. |
- 2026-10-20 19:40 | R18 | fam 1/2 (transcript-level TIS competition from the cap = r15_tisnolocal_crop): seeds 1-2, and weight 300 on TIS crop multinomial + CHX local | configs/r18/* | 11172655 | txtis 3 seeds 74.7+-1.0 (75.8/73.8/74.6); txtis300 74.7; all pass | NO GAIN over the lall300 leader (75.2+-0.2); +1.4 over the same-weight base (inside noise). Cap-anchored TIS competition = 1 kb block competition. Streak 1 (fams 1,2). |
- 2026-10-20 21:00 | R19 | fam 4 (noise model) on the lall300 leader: + NB per-nt (1, seeds 0-1; 10), + Poisson per-nt (10) | configs/r19/* | 11173142, 11173143 | nb1 2 seeds 74.7 (74.1/75.3); nb10 74.9; poisson_nt10 74.9; all pass | NO GAIN (streak 2: fam 4). Per-nt NB / Poisson likelihoods add nothing to the multinomial-profile + Poisson-total decomposition. |
- 2026-10-20 21:40 | R20 | fam 3 (coupling 0.5), fam 6 (CHX frame 50; separate CHX trunk conv25x2 for longer-range periodicity), fam 5 (minimal inputs: embedding + sense RNA only) on the lall300 leader | configs/r20/* | 11173155, 11173156 | coup05 74.8; chxframe50 74.8; sense-only 74.0; chx_trunk conv25x2 75.8 (CHX frame .69, codon 67.6, nt 62.1); all pass | NO PRIMARY GAIN (streak 3 across fams 1,2,4,3,5,6) -> STOP RULE MET. Separate wide CHX trunk confirmed (11173159, 11173160): 3 seeds primary 75.6+-0.2 (75.8/75.4/75.4) = leader on TIS; CHX nt 61.1 (+2.4), frame .68 (+.04), codon 66.9. Recommended recipe. |
