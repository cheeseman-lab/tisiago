# P6 fast search ledger

## Final summary
(pending)

## Leaderboard
| name | primary | K562 TIS codon | nt | guards | seeds |
|---|---|---|---|---|---|
| r01_foc_lall20 | 64.9 | 66.0 | 63.9 | pass (CHX bin30 74.9 codon 52.1) | 1 |
| r01_plain_lw20 | 53.5 | 55.8 | 51.2 | pass | 1 |
| ref_rnacond_ps | 42.5 (2 seeds) | 45.7 | 39.3 | ref | 2 |
| (fail) r01_foc_lw50 | 67.8 | 69.0 | 66.5 | FAIL CHX | 1 |
| (fail) foc_lw5 | 65.9+-1.0 | 66.8 | 64.4 | FAIL CHX | 3 |

## Findings
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
- 2026-10-20 15:19 | R08 | fam 5 (conv depth): conv9x2 seeds 1-2, conv9x3, conv9x4, conv5x3, conv15x2, conv9x2 hidden 128 / 512 | configs/r08/* | 11170968, 11170969 | pending |
- 2026-10-20 16:27 | R09 | fam 5 + fam 6 (separate wide CHX trunk for elongation/frame) + optimisation on conv9x2 base: 30k steps; chx_trunk conv51; chx_trunk conv21x2; no CHX contrast | configs/r09/* | 11171706 | pending |
