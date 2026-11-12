# tisiago — plan and status

**Updated:** 2026-11-12 · Single entry point for what we are doing and why.

Method detail for the next phases: [`specs/2026-11-11-rescope-native-caller.md`](specs/2026-11-11-rescope-native-caller.md)
(the re-scope spec). What transfers to a new recipe: [`PRINCIPLES.md`](PRINCIPLES.md). Results:
[`../FINDINGS.md`](../FINDINGS.md). Hand-off from the label pipeline: [`DATA_CONTRACT.md`](DATA_CONTRACT.md).

---

## 1. Naming

Names say what a thing is, not when it was tried. Retired names (P1-P6, Option B, AR, experiments 1-3,
steps 1-2, classes 1-3, `seq2func`, `rnacond`, `sbflux_<g>`) survive only in FINDINGS § titles and the
`archive/pre-cleanup-2026-11-12` tag.

### Lines of work

| Name | What | Was |
|---|---|---|
| **tracks** | Fine-tuned AlphaGenome, RNA-conditioned, predicting harringtonine (TIS) and CHX P-site tracks; one line held out per fold | P6, seq2func, class 3 |
| **calling** | Tracks -> start calls: the caller-free score (`calling.track_score`), the native caller (Phase 3) | Ribo-TISH on synthetic BAMs, `track_tis_eval` |
| **truth** | Reproducible starts from the label pipeline, tagged by codon class, region, pause | (new) |
| **baseline** | Frozen AlphaGenome embeddings + logistic head, one label per codon — the comparator | class 1, P1/P2/Option B |
| **floor** | One-hot codon / ±20 nt Kozak + the same head — the bar every model must clear | "one-hot ±20 bp floor" |

Retired lines, recoverable from the tag: Evo2 (all arms), per-condition track models (class 2), the
synthetic-BAM round trip, tree heads, the P4 efficiency regression.

### Tracks runs

Four axes, always in this order: `tracks.<stage>.<chx>.<cond>.ho-<line>.s<seed>`.

| Axis | Values | Was |
|---|---|---|
| stage | `genomic` — fine-tuned on genomic windows, head in genome space · `spliced` — warm-started from `genomic`, fine-tuned on the spliced mRNA, head in transcript space | step 1 / `rnacond_fc2d` · step 2 / `txtrain` |
| chx | `free` — CHX per position · `flux` — predicted initiation propagated in frame, stopped at stop codons | (default) · `chx_flux` replace |
| cond | `pos` — positional RNA into the head · `film` — global regulator vector, FiLM, learned · `corgi` — Corgi's pretrained pathway, frozen · suffix `@head` / `@trunk` | only `pos` exists |
| holdout | `K562`, `HeLa`, `U2OS`, `RPE1` (RPE1 async / quiescent / senescent are one group) | `rnacond:<G>`, `<g>` |

**Reference recipe:** `tracks.spliced.flux.pos` — `configs/tracks/genomic_fc2d.args` for stage 1,
`configs/tracks/spliced_flux_pos.yaml` for stage 2. Phase 1 re-baselines exactly this on the clean data.

Baseline arms: `baseline.ag` (AlphaGenome 16 kb + 131 kb + Kozak ±20), `floor.kozak20`, `floor.codon12`.

---

## 2. Where we are

| Line | Status | Headline |
|---|---|---|
| baseline | Closed; rerun once on clean labels as the comparator | AG + Evo2 recall 0.306 @ 0.99 FP/transcript, clean protocol (§10). AG-only: 0.225 @ ≤1 FP/transcript (§7), single seed with a test-chosen threshold — no clean-protocol AG-only number exists; Phase 1 produces it |
| tracks | Live | `tracks.spliced.flux.pos`: TIS codon 70-106% of replicate by line; uORF framing doubled (§15-16) |
| calling | Only through Ribo-TISH on synthetic BAMs, now retired | Jaccard 0.39 K562 (deployed cutoff, §16); caller-free score never run on tracks |
| truth | Lands 2026-11-13 | — |

Tracks beats per-condition models on track metrics. **Whether it beats the baseline at calling starts is
unmeasured** — Phase 1 measures it.

Not adopted after §16: TIS loss reweighting, codon-aware dwell, harringtonine leak (§17).

---

## 3. Upstream input (owned outside this plan)

- Ribo-TISH (fork) calls starts over the matched short-read mRNA.
- Long-read data, RNA and the replicates refine the call set.
- Long-read and CAGE data filter to **strong starts, mostly replicated between cell lines**.
- That set is also the hand-off to swissisoform (protein products, mutation and disease).

The truth set in Phase 0 is that filtered, cross-line-replicated set, not raw Ribo-TISH output. It changes
the denominators of every start metric in FINDINGS §11-17.

---

## 4. Objectives

Two primary objectives, then a break. Everything after the break needs a model that already works.

### Primary — a model that works out of distribution

- **O1 — decent starts in a line never trained on.** Given only RNA-seq, a start set comparable to the
  label pipeline's for that line. *Done when* M1 on a held-out line beats `baseline.ag`, with the threshold
  fit only on training lines.
- **O2 — a line outside the four.** Leave-one-line-out across four backgrounds can still be fitting to our
  own four lines as we prepare them; a fifth line is the real out-of-distribution test, and it needs RNA-seq
  only. Treat this as part of the primary result, not a follow-up. *Done when* M1 there is within noise of O1.

### Interpretation and validation — after O1 holds

Deferred deliberately. Interpreting a model that does not yet work spends effort explaining predictions that
may not track reality.

- **O3 — variant effects.** An upstream stop should suppress a downstream start. **This is a sanity check at
  the interpretation stage, not evidence of model quality:** a human can write that rule from first
  principles, so the model agreeing tells us little about whether the model is good. Its value is guarding
  attribution once there is something worth attributing. Known-effect variants first (Kozak context, uORF
  start gain/loss, uAUG-creating 5'UTR variants).
- **O4 — what drives differential start use between lines.** Attribution on the conditioning pathway; needs
  O1 and M2. Cleanest substrate: the RPE1 triplet (three cell states, one genotype).
- **O5 — novel starts in an unprofiled line, validated by N-terminal proteomics.** Needs O1, O2 and a search
  pipeline. The one validation signal independent of Ribo-seq and of any caller, and so the only clean way to
  separate a real alternative start from an elongation pause (§17).
- **O6 — synthesis.** How alternative initiation is sequence-encoded, and how the transcript layer shifts the
  canonical : alternative : non-canonical ratio. The publication frame, not a work item.

## 5. Phases

Two tracks. The **data track** waits on the truth set; the **conditioning track** needs no new data and runs
from day one.

### Now, before the data lands

| Phase | What | Cost |
|---|---|---|
| **0a** | Run `calling.track_score` on the current `tracks.spliced.flux.pos` exports against `baseline.ag`, on the **existing** labels. Both arms are already measured on those labels, so the comparison is internally consistent and available today. Answers whether `tracks` is ahead of or behind the baseline at calling starts — open since §12. | hours, CPU |
| **0b** | Assert the protocol guardrails before any agent-driven hill climb: that `--i-am-final` is *enforced* and not merely a flag; that no fold warm-starts from a checkpoint that saw its held-out line (a two-stage provenance chain, not one check); that percent-of-replicate uses the identical transcript set for model and replicate. An optimiser will reach test data if a path exists. | small, CPU |

### Data track

| Phase | What | Cost |
|---|---|---|
| 0 | Data contract, truth set, P-site convention assertion, differential-start counts | small, CPU |
| 1 | Re-baseline `tracks.spliced.flux.pos` unchanged on clean data; rerun `calling.track_score`, `baseline.ag` and the floors on clean labels | 4 folds x 3 seeds, GPU |
| 2 | M1 / M2 / M3 into the scorecard | small, CPU |
| 3 | Native caller + learned threshold; three-way decomposition (Ribo-TISH on real BAMs / native on real tracks / native on predicted) | small, fast loop |
| 4 | Architecture work, driven by what M1 / M2 say limits O1 | the bulk |

### Conditioning track — parallel, no new data

`film`, `corgi` and `@trunk`, per [issue #1](https://github.com/cheeseman-lab/tisiago/issues/1). Moved out of
Phase 4: decision 4 establishes that backbone conditioning was never tested, so this is unexplored ground on
O1's critical path rather than late-stage architecture work. Note that the fast loop **cannot** judge
`@trunk` — it runs on a frozen embedding cache — so schedule those as full fine-tunes from the start instead
of discovering the gap later.

### Deferred

Proteomics validation (O5) and attribution / in-silico mutagenesis (O3, O4) follow O1. See §4.

**Next month:** 0a, 0b, 0, 1, 2, 3 and the conditioning track. Not O3, not attribution.

**Critical path:** 0a and 0b now -> truth lands -> 0 -> 1 -> 2 -> 3, with conditioning alongside throughout.

## 6. How the next month runs

- **Measure before modelling.** Phase 0a is first because it can change the plan. If `tracks` loses to the
  baseline at calling starts, that is the most important finding available and it should be known in week one,
  not week four.
- **Freeze the metric before looking at output.** M1, M2 and the truth set are committed and tested before the
  first training run. What cost the last stretch was climbing metrics chosen for convenience —
  percent-of-replicate and synthetic-BAM Jaccard — rather than metrics that matched the objective.
- **Keep a ledger with an explicit stop rule.** One row per config: what changed, the validation number, the
  decision. The retired `p6fast` ledger used "no gain > 2 across 3 consecutive families -> STOP"; keep that
  shape. Without it, a month of iteration yields many configs and no conclusion — and that ledger is the only
  reason decision 4 could be answered at all.
- **Mark untested claims as untested.** "RNA into the head, not the backbone" read as a finding for months
  while being a plumbing decision, and it closed off a direction nobody revisited. Write what was measured.
- **Decision gate at the halfway point.** If M1 on a held-out line has not beaten `baseline.ag`, stop adding
  architecture and go back to diagnosis (PRINCIPLES §6, "diagnose before building").
- **If the data slips, cut Phase 4 — never Phase 0a.** The measurement is what says whether the rest is worth
  doing, and it does not depend on the new data.

## 7. Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Baseline: AG + Evo2 (`ag7`) or AG-only | **Resolved 2026-11-12: AG-only.** The comparator drops from ~0.30 to ~0.225 recall (§7, oracle threshold); say so wherever it is quoted, so `tracks` is not credited for a weaker bar. |
| 2 | Calibration target for the clean-cascade cutoffs ([ribotish#14](https://github.com/cheeseman-lab/ribotish/issues/14)) | Open |
| 3 | AlphaGenome 2 as a backbone | Conditional on weights; a Phase 4 controlled swap (AG2 has no Ribo-seq output) |
| 4 | Was "RNA into the head, not the backbone" tested? | **Resolved 2026-11-12: no** — only head-side feature variants were; `@trunk` is new ground |

---

## 8. Pitfalls carried forward

- **Do not climb percent-of-replicate or synthetic-BAM Jaccard.** The first inflates when replicates
  disagree (RPE1 reads 179%); the second has a no-model ceiling of 0.85-0.89.
- **Separate initiation from pausing.** 83% of reproducible later-exon near-cognate sites are in-frame inside
  the CDS, AGG 2.5x enriched (§17), corroborated by inhibitor-free TISCA2 (Ichihara et al. 2026).
- **Peak sharpness is a per-library property**, not predictable from RNA.
- **Three training lines.** Prefer a pretrained conditioning pathway over fitting one here (#1).
- **Score test once**, after validation decisions, behind `--i-am-final`.
