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

Ordered; each rung needs the one before it.

- **O1 — decent starts in a line never trained on.** Given only RNA-seq, a start set comparable to the
  label pipeline's for that line. *Done when* M1 on a held-out line beats `baseline.ag` with the threshold
  fit only on training lines. **The preliminary objective.**
- **O2 — transfer to a further line.** RNA-seq only. *Done when* M1 there is within noise of O1.
- **O3 — variant effects (cheap, start early).** An upstream stop should suppress a downstream start.
  Known-effect variants first (Kozak, uORF start gain/loss, uAUG-creating 5'UTR variants); a small curated
  set on the baseline too, since it needs re-extraction per variant.
- **O4 — what drives differential start use between lines.** Attribution on the conditioning pathway; needs
  O1, O3's sanity check and M2. Cleanest substrate: the RPE1 triplet.
- **O5 — novel starts in an unprofiled line, validated by N-terminal proteomics.** Needs O1, O2 and a search
  pipeline.
- **O6 — synthesis.** How alternative initiation is sequence-encoded, and how the transcript layer shifts the
  canonical : alternative : non-canonical ratio. The publication frame, not a work item.

---

## 5. Phases (spec §10)

| Phase | What | Cost |
|---|---|---|
| 0 | Data contract, truth set, P-site convention assertion, differential-start counts | small, CPU |
| 1 | Re-baseline `tracks.spliced.flux.pos` unchanged on clean data; run `calling.track_score` on it (never run); rerun `baseline.ag` and the floors on clean labels | 4 folds x 3 seeds, GPU |
| 2 | M1 / M2 / M3 into the scorecard | small, CPU |
| 3 | Native caller + learned threshold; three-way decomposition (Ribo-TISH on real BAMs / native on real tracks / native on predicted) | small, fast loop |
| 4 | Architecture work driven by what M1/M2 say limits them; conditioning (`film`, `corgi`, `@trunk`) per issue #1 | the bulk |
| 5 | Proteomics validation | small, needs a search pipeline |
| 6 | Attribution and in-silico mutagenesis, known-effect variants first | moderate |

**Next month:** Phases 0-3, the conditioning experiments (#1, no new data needed), O3's known-effect check.

**Critical path:** truth lands -> Phase 0 -> Phase 1 -> Phase 2 -> Phase 3.

---

## 6. Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Baseline: AG + Evo2 (`ag7`) or AG-only | **Resolved 2026-11-12: AG-only.** The comparator drops from ~0.30 to ~0.225 recall (§7, oracle threshold); say so wherever it is quoted, so `tracks` is not credited for a weaker bar. |
| 2 | Calibration target for the clean-cascade cutoffs ([ribotish#14](https://github.com/cheeseman-lab/ribotish/issues/14)) | Open |
| 3 | AlphaGenome 2 as a backbone | Conditional on weights; a Phase 4 controlled swap (AG2 has no Ribo-seq output) |
| 4 | Was "RNA into the head, not the backbone" tested? | **Resolved 2026-11-12: no** — only head-side feature variants were; `@trunk` is new ground |

---

## 7. Pitfalls carried forward

- **Do not climb percent-of-replicate or synthetic-BAM Jaccard.** The first inflates when replicates
  disagree (RPE1 reads 179%); the second has a no-model ceiling of 0.85-0.89.
- **Separate initiation from pausing.** 83% of reproducible later-exon near-cognate sites are in-frame inside
  the CDS, AGG 2.5x enriched (§17), corroborated by inhibitor-free TISCA2 (Ichihara et al. 2026).
- **Peak sharpness is a per-library property**, not predictable from RNA.
- **Three training lines.** Prefer a pretrained conditioning pathway over fitting one here (#1).
- **Score test once**, after validation decisions, behind `--i-am-final`.
