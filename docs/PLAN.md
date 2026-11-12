# P6 project plan — from RNA-seq of an unprofiled cell line to a sequence model of alternative initiation

**Date:** 2026-11-12 · **Status:** proposed · Single entry point for the next phase of work.

Method detail lives in [`superpowers/specs/2026-11-11-p6-rescope-native-caller-design.md`](superpowers/specs/2026-11-11-p6-rescope-native-caller-design.md)
(the re-scope spec) and [`P6_PRINCIPLES.md`](P6_PRINCIPLES.md) (what transfers). Results are in
[`../FINDINGS.md`](../FINDINGS.md); live status in [`../ROADMAP.md`](../ROADMAP.md). This document holds the
**arc and the sequencing** — what we are trying to establish, in what order, and what counts as done.

---

## 1. Where we are

Three classes of experiment (spec §0 states this in full):

| Class | What | Status |
|---|---|---|
| **1** | Frozen embeddings -> direct start calls (FINDINGS §1-10) | **Closed. The baseline.** recall 0.306 @ 0.99 FP/transcript; AUG 0.58 vs near-cognate 0.07 |
| **2** | Fine-tuned AlphaGenome -> tracks, per condition, DNA only (§11-12) | **Abandoned.** Starts 0.035-0.099 vs the baseline's 0.306; no cross-line transfer |
| **3** | Fine-tuned AlphaGenome -> tracks, one shared RNA-conditioned model, leave-one-cell-line-out (§13-17) | **Live. Build from here.** TIS codon 70-106% of replicate by line |

Class 3's advantage over class 1 is established on **track** metrics. Whether it wins at **calling starts** is
unmeasured — that is the first thing Phase 1 of the spec does.

---

## 2. Upstream input (owned outside this plan)

The label pipeline is being finalised separately and is **not a deliverable here**. In outline, so that
downstream work can be written against it:

- Ribo-TISH is the caller, over the matched short-read mRNA it uses.
- Long-read data plus the RNA and the replicates refine the call set.
- Long-read and CAGE data (in-house where it exists, ENCODE otherwise) then filter down to a set of
  **strong starts, mostly replicated between cell lines**.
- That filtered set is the hand-off to **swissisoform**, which characterises the protein products of these
  transcripts and maps them to mutation and disease.

**Consequence for us:** the truth set in spec Phase 0 is this filtered, cross-line-replicated set, not raw
Ribo-TISH output. That is a stronger truth set than the one FINDINGS §11-17 used, and it changes the
denominators. Phase 0 starts when it lands; nothing before then depends on it.

---

## 3. Objective ladder

Ordered. Each rung needs the one before it to hold.

### O1 — primary: decent starts in a cell line we never trained on

Given only RNA-seq from a held-out line, produce a start set comparable to what the label pipeline produces
for that line.

- **Needs:** spec Phases 0-3 (data contract, re-baseline, re-scoped metrics, native caller with a learned
  threshold), plus the conditioning work in [#1](https://github.com/cheeseman-lab/tisiago/issues/1).
- **Done when:** M1 (precision/recall against reproducible starts at a fixed per-transcript call budget, on a
  held-out line, split AUG vs near-cognate and in-CDS vs outside) beats the class-1 baseline, with the
  threshold fitted only on training lines.
- **This is the whole preliminary objective.** Everything below waits on it.

### O2 — transfer to a further cell line

The same model on a line outside the current four. Tests whether O1 generalised or fitted our four backgrounds.

- **Needs:** O1, plus RNA-seq for the new line. No new Ribo-seq.
- **Done when:** M1 on the new line is within noise of the held-out-line result in O1.

### O3 — variant effects (cheap; can start early)

A mutation that introduces a stop codon upstream should suppress use of a downstream start. Predicted
direction should match.

- **Needs:** nothing new. This is in-silico mutagenesis on an existing model.
- **Worth doing on the class-1 baseline too**, as the user noted — it gives a comparator. Note the cost
  asymmetry: the fine-tuned track model yields gradients directly, whereas the frozen pipeline needs
  embedding re-extraction per variant, so keep the frozen arm to a small curated variant set.
- **Start with variants of known effect** — Kozak-context changes, uORF start gain/loss, 5'UTR variants
  creating a uAUG — before anything novel. If predicted direction does not match there, attribution on this
  model cannot be trusted.
- **Done when:** predicted direction matches on a curated known-effect set, reported per variant class.

### O4 — what drives differential start use between lines

Attribution and ablation on the conditioning pathway: which sequence features, and which trans-regulatory
context, explain a start being used in one line and not another.

- **Needs:** O1 and O3's sanity check. Also M2 from the spec (does the model get the *direction* of
  reproducible between-condition changes right?) — without M2, attribution explains predictions that may not
  track reality.
- **Cleanest substrate:** the RPE1 async/quiescent/senescent triplet — three cell states, one genotype.

### O5 — novel starts in an unprofiled line, validated off-sequencing

Predict starts in a line with no initiation profiling, then confirm some experimentally.

- **Needs:** O1, O2.
- **Validation signal:** N-terminal proteomics, which establishes a start as real because it produced a
  protein with that N-terminus — independent of Ribo-seq and of any caller. This is the discriminator
  FINDINGS §17 needs, since most in-CDS in-frame near-cognate "starts" there are elongation pauses. Public
  HeLa/K562 N-terminomics suffices to start; other cell lines extend it.
- **Done when:** a predicted-novel set in an unprofiled line has proteomic support above a matched control
  set.

### O6 — synthesis: how alternative initiation is encoded in sequence

The claim the ladder is built to support, in two parts:

1. **Sequence:** what in the genomic sequence determines alternative and non-canonical initiation, from
   ablation of a model strong enough to interrogate.
2. **RNA biology:** how which transcripts exist alters the ratio of canonical to alternative to
   non-canonical starts — i.e. the transcriptional layer feeding the initiation layer.

- **Needs:** O1-O5.
- Likely beyond the current scope and worth treating as the publication frame rather than a work item.

---

## 4. Scope for the next month

**In:** spec Phases 0-3 (O1), the conditioning experiments in #1, and O3's known-effect variant check, which
is cheap and de-risks O4 early.

**Out:** O2, O4, O5, O6. O5 needs a proteomics search pipeline that does not exist yet; O4 needs M2 to be
measurable, which Phase 0 item 5 checks.

**Critical path:** label pipeline lands -> Phase 0 -> Phase 1 (including the unmeasured class-3 start number)
-> Phase 2 metrics -> Phase 3 caller. The conditioning work runs in parallel from the start, since it needs
no new data.

---

## 5. Open decisions

| # | Decision | Why it matters |
|---|---|---|
| 1 | Does the clean-data class-1 baseline stay `ag7`, or drop to AG-only? | Dropping Evo2 weakens the comparator 0.300 -> 0.225, so class 3 looks better without improving. Evo2 appears in **zero** files under `src/tisiago/seq2func/` — dropping it simplifies only the baseline, not class 3. Changes spec Phase 1. |
| 2 | Which calibration target do the clean-cascade cutoffs have to respect? | Raised in [ribotish#14](https://github.com/cheeseman-lab/ribotish/issues/14): the cutoffs are global constants over libraries that differ in depth and periodicity. Needs the assay and the quantity named. |
| 3 | AlphaGenome 2 as a backbone | Conditional on weights shipping; a Phase 4 controlled swap, never a drop-in (AG2 emits no ribosome-profiling track). See #1. |

**Resolved 2026-11-12:** whether `P6_PRINCIPLES.md` §2's "RNA into the head, not the backbone" was tested.
It was **not**. The cited negatives (CPM, transcript-relative, one-sided/antisense, 5'-end sharpness) are all
head-side *feature* variants, and the fast loop runs on a frozen embedding cache, so backbone conditioning is
not expressible in it (`autoresearch/p6fast/ledger.md`; principles §6 says as much). Trunk conditioning is new
ground, not a repeat.

---

## 6. Pitfalls carried forward

- **Do not climb percent-of-replicate or synthetic-BAM Jaccard.** The first inflates when replicates disagree
  (RPE1 reads 179%); the second measures a lossy adapter whose own ceiling is 0.85-0.89 with no model involved.
- **Separate initiation from pausing.** 83% of reproducible later-exon near-cognate sites are in-frame inside
  the CDS, AGG 2.5x enriched — elongation pauses, corroborated independently by the inhibitor-free TISCA2 data
  in Ichihara et al. 2026. Never count them as plain positives.
- **Peak sharpness is a per-library property** and is not predictable from RNA. A held-out line gets the
  training lines' average; compare against a replicate at equal depth.
- **Three training lines.** Every published cell-state-conditioning model trains across many cell types; we
  have 4 genetic backgrounds. Prefer borrowing a pretrained conditioning pathway over fitting one here (#1).
- **Score test once**, after validation decisions, behind the `--i-am-final` gate.
