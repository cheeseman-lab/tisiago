# Promotion: conv9 trunk + local multinomial on both tracks (weight 100)

Status: meets the promotion rule (2026-10-20 16:30). Superseded candidates may follow (conv9x2, see ledger).

## Config (fast loop; `autoresearch/p6fast/configs/r06/r06_conv9_s1.yaml`, seeds 0/1/2 = r04_lall100_conv9, r06_conv9_s1, r06_conv9_s2)
```yaml
inputs: [embedding, rna, rna_onesided]
model:
  trunk: {kind: conv, hidden: 256, layers: 1, width: 9}
  chx_contrast: true            # dispensable: r07_conv9_nocontrast 70.6 (same)
  tis: {kind: factorised, block: 1024, two_grids: true}
  chx: {kind: softplus}
losses:
  - {kind: poisson, scope: sample, weight: 1.0}
  - {kind: multinomial, scope: local, block: 1024, libraries: [tis, chx], weight: 100}
  - {kind: multinomial, scope: global, weight: 5.0}
```

## Scores (K562 held out, val chr7, percent of replicate; 3 seeds, mean +- sd)
| | primary | TIS codon | TIS nt | TIS bin30 | CHX bin30 | CHX codon | CHX nt | CHX frame |
|---|---|---|---|---|---|---|---|---|
| conv9 (this) | **70.6 +- 0.3** | 71.7 | 69.4 | 84.7 | 78.2 | 57.7 | 47.6 | 0.60 |
| slow leader fast equivalent (foc_lw20; r01_foc_lw20, r07_foc_lw20_s1/s2) | 66.8 +- 0.3 | 68.0 | 65.6 | 83.9 | 71.0 (FAIL) | 46.4 (FAIL) | 33.9 (FAIL) | 0.49 |
| ref_rnacond_ps (2 seeds) | 42.5 | 45.7 | 39.3 | 69.7 | 73.4 | 49.9 | 38.3 | 0.51 |

Gain over the slow leader's fast equivalent: +3.8 primary, and the CHX guards that foc_lw20 fails
in the fast loop now pass with margin (CHX codon +7.8 over ref).

## Rationale (biology)
- Local (1 kb) multinomial on TIS **and** CHX: ribosome density is compositional within a
  transcript region: initiation sites compete for scanning ribosomes, and elongating ribosomes
  are distributed along the local ORF. Putting it on CHX removes the CHX cost of the TIS-only loss.
- Width-9 convolution over the embedding: the start codon and its Kozak-scale neighbourhood
  (about +-4 nt) and the P-site frame are read from neighbouring nucleotides, which a per-nt
  head cannot see (conv3 68.0 < conv9 70.6; per-nt MLP 67.1).

## Mapping to the slow tier (rnacond, `src/tisiago/seq2func/train.py` / `model.py`)
- Exists: factorised TIS output + one-sided RNA + CHX contrast = `--head rnacond_foc`;
  local loss on both tracks = `--local-weight 100 --local-libraries tis,chx --local-block 1024`.
- Missing: a `rnacond_foc` variant whose `hidden` layer is `Conv1D(256, kernel_shape=9)` instead of
  `Linear(256)` (the base `rnacond` head already uses a Conv1D(256, 31); foc replaced it with a
  Linear). Suggested name `rnacond_focc9` (or `rnacond_fc9` without the CHX contrast, which the
  fast loop shows is dispensable).
- Caveat: the fast loop's Poisson term is on per-transcript totals (`scope: sample`); the slow tier
  uses genomic-window totals. The local weight of 100 was tuned against that Poisson scale; in the
  slow tier start from `--local-weight 20` and `100` as two arms.
