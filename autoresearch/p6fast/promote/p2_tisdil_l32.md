# Promotion (phase 2): TIS-only dilated context + 32-nt local TIS competition on the phase-1 recipe

Status: final phase-2 recommendation (stop rule met 2026-10-21 18:25; see ledger "Phase 2 summary"). Beats the phase-1 leader on primary2 by +9.8 (3-seed means, threshold 5), guards intact.

## Config (`autoresearch/p6fast/configs/p2r04/p2r04_tisdil_l32.yaml`, `configs/p2r06/p2r04_tisdil_l32_s{1,2}.yaml`)
```yaml
train: {steps: 30000}
inputs: [embedding, rna, rna_onesided]
model:
  trunk: {kind: conv, hidden: 256, layers: 2, width: 9}          # shared trunk
  tis_context:                                                    # NEW: TIS branch only
    {kind: dilated, hidden: 256, channels: 128, dilations: [1, 2, 4, 8, 16, 32, 64, 128, 256]}
  chx_trunk: {kind: conv, hidden: 256, layers: 2, width: 25}
  chx_contrast: false
  tis: {kind: factorised, block: 1024, two_grids: true}
  chx: {kind: factorised, block: 1024, two_grids: true}
losses:
  - {kind: poisson, scope: sample, weight: 1.0}
  - {kind: multinomial, scope: local, block: 1024, libraries: [tis, chx], weight: 300}
  - {kind: multinomial, scope: global, weight: 5.0}
  - {kind: multinomial, scope: local, block: 32, libraries: [tis], weight: 300}   # NEW
```

## Scores (K562 held out, val chr7; 3 seeds; peak values as fractions, real in brackets)
| | primary2 | primary (ph1) | TIS codon | TIS nt | CHX bin30 / codon / nt / frame | nc argmax0 [0.311] | nc meta [0.170] | ATG argmax0 [0.655] | ATG meta [0.331] |
|---|---|---|---|---|---|---|---|---|---|
| this config | **70.4 +- 1.3** | 81.2 | 81.8 | 80.5 | 82.1 / 67.2 / 61.8 / .68 | 0.188 | 0.075 | 0.64 | 0.20 |
| TIS dilated context only | 66.1 +- 1.5 | 78.4 | 79.8 | 77.1 | 82.0 / 67.4 / 62.1 / .69 | 0.171 | 0.068 | 0.68 | 0.19 |
| phase-1 leader r20_l300_chx25x2 | 60.6 +- 0.3 | 75.4 | 76.4 | 74.7 | 82.2 / 66.5 / 60.6 / .68 | 0.145 | 0.058 | 0.70 | 0.19 |

Guards: phase-1 guards vs ref_rnacond_ps pass; TIS codon/nt are +5.4/+5.8 above the phase-1 leader (floor -2).
Near-cognate per-site argmax at the start rises 0.145 -> 0.188 (47% -> 60% of real); ATG drops slightly
(0.70 -> 0.64, still at real level 0.66).

## Rationale (one line each)
- TIS-only residual dilated context (RF ~1 kb): whether a weak near-cognate codon is used depends on its
  surroundings over a few hundred nt (position in the 5'UTR relative to the cap and to the next ATG,
  uORF / leaky-scanning context), which a 17-nt conv cannot see; elongation (CHX) keeps its own trunk.
- 32-nt local TIS multinomial: alternative starts lie within a few codons of each other or of the main
  start; competition at codon scale forces the peak onto the exact initiating codon instead of
  smearing it over the neighbourhood (the 1 kb block alone only places mass in the right region).
- Dissection (R05/R07, single seeds): range ~130 nt gives most of the context gain alone (65.5) but with
  the 32-nt term 1 kb is better (70.4 vs 67.7); an undilated residual stack (RF ~19) gives 64.4, and
  wide plain conv (62.1) / block attention (63.8) help less.

## Mapping to the slow tier (`src/tisiago/seq2func/model.py`, `train.py`)
- Exists: `rnacond_fc2` = conv9x2 shared trunk, factorised TIS + CHX, one-sided RNA, no contrast;
  `_dilated_trunk` (residual kernel-3 dilated stack, zero-initialised branches) used by `rnacond_fod`.
- Missing:
  - New head kind (e.g. `rnacond_fc2d`): `rnacond_fc2` with `tis_in = _dilated_trunk(Linear(256)+relu(h))`
    for the TIS output only (CHX output on `h`), dilations 1..256 (the fast loop also applies a
    Linear(256)+relu before the residual stack). The fast loop's separate CHX conv25x2 trunk is also not in
    `rnacond_fc2` (phase-1 recipe; CHX only).
  - A second local multinomial term in `loss.poisson_multinomial` / train flags (e.g.
    `--local2-block 32 --local2-weight 300 --local2-libraries tis`); the slow tier has one local block.
- Weights: the slow tier's Poisson is on genomic-window totals; also run the 32-nt term at 100.
