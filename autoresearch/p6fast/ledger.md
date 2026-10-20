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
- Fam 1: local (1 kb) multinomial on TIS gives the start-peak gain; its weight 5->50 adds only ~2 points in the fast loop. Adding the same local multinomial on CHX removes the CHX cost and lifts CHX above ref (bio: elongating ribosome density is shaped within a transcript region, so both tracks are compositional within ~1 kb, not just in total).
- Seed check: t3_foc_lw5_ps fails the K562 CHX guards (bin30 70.4 vs 73.6; codon 45.8 vs 50.1; nt 33.5 vs 38.7). Fixing the CHX cost of the factorised/local TIS head is a precondition for any winner.

## Rounds
- 2026-10-20 ~13:00 | R01 | fam 1 (competition) + noise floor | reproduce local weight 20; sweep 10/50; local on CHX too; plain softplus + local 20; seeds 1-2 of foc_lw5 and ref | configs/r01/* | jobs 11167834 (lw20, lw10, lw50, lall20), 11167835 (lw5_s1, lw5_s2, ref_s1, plain_lw20) | foc_lw5 3 seeds 65.9+-1.0 (64.9/65.0/66.9); lw10 66.3, lw20 66.7, lw50 67.8 (all FAIL CHX guards: bin30 ~71, codon ~46); lall20 64.9 PASS (CHX bin30 74.9, codon 52.1, nt 40.4 > ref); plain_lw20 53.5 pass; ref_s1 42.9 (ref 42.1) | lw20 reproduces only weakly in the fast loop (+0.8 over lw5, inside noise). Guarded leader = foc_lall20. Next: lall weights up, split CHX trunk. Added model.split_trunk (+tests). |
- 2026-10-20 ~13:50 | R02 | fam 1 (competition on both tracks) + fam 5 (split trunk) | does local CHX multinomial let TIS local weight rise without CHX cost; is the CHX cost from the shared trunk | configs/r02/* (lall50, lall100, t50c20, lall20_s1/s2, lw50_split, lall20_split, lall50_split) | 11167876, 11167877 | pending |
