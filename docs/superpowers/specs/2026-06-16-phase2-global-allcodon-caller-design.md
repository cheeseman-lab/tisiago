# Phase 2 — Global all-codon calibrated caller

**Date:** 2026-06-16
**Status:** design, pre-implementation
**Depends on:** Phase 1 (`src/tisiago/caller.py`, merged) — provides the calibrated head + metric functions this phase applies.
**Direction:** `CLAUDE.md` → *Direction*, build step "Phase 2 — global all-codon calibrated caller".

## Goal

Turn the curated candidate-ranker into a **dense caller**: enumerate *every* codon in
held-out expressed transcripts, embed them, apply the Phase-1 calibrated head, and report
the headline metric **at true genome-wide imbalance** plus the **non-cognate ≈ 0**
grounding control. This is the deliverable the PoC was a stepping stone toward.

## Key insight — most of this already exists

The store design *decouples the forward-pass window from the sliced positions*, so a dense
scan reuses the existing pipeline almost wholesale:

- **`tiling.group_into_tiles`** already takes arbitrary candidate rows `(row_idx, chrom,
  gstart, strand, codon)` and groups them into shared forward-pass tiles. Dense enumeration
  just produces *more rows per transcript* — the same tiles, slicing hundreds of positions
  instead of ~4. Forward-pass count scales with genomic span ÷ step, **not** with the number
  of enumerated positions.
- **`extract.py`** already reads a manifest parquet, shards by `transcript_id`, tiles,
  embeds, and writes per-shard `.npz`. It runs unchanged on a *scan manifest*.
- **`store.py`** already assembles shards into a row-aligned store.
- **`caller.py`** already trains+calibrates a head and computes `recall_at_fp_budget` /
  `reliability` from plain arrays.

So Phase 2 adds exactly **one new module** (GTF → scan manifest) and **one new evaluation
entry** (apply the head trained on the curated store to the dense scan store at true
imbalance). The GPU step is the existing `extract.py` pointed at the scan manifest.

## Resolved design decisions

1. **Scope:** enumerate + embed the held-out **`test` (chr8/chr9) and `val` (chr7)**
   expressed transcripts only (~2,272 transcripts), not the whole genome. That is all the
   headline number needs; genome-wide deployment is a later, mechanical scale-up.
2. **Persist a scan store** (reuse `store.py` layout, separate dir `data/scan_store/`) so
   head experiments stay GPU-free, exactly like today. Bounded by the held-out transcript
   count.
3. **Position model:** enumerate *every* transcript position `0..L-3`, read the 3-mer
   starting there, classify `codon_class ∈ {AUG, near_cognate, non_cognate}`. "All codons"
   = every reading position, every frame (initiation can occur in any frame).

## Components

### New: `src/tisiago/enumerate_codons.py` (CPU, fully testable)

GTF-driven enumeration → a **scan manifest** parquet, schema-compatible with the candidate
manifest so `extract.py` consumes it unchanged.

- **Inputs:** GENCODE v49 GTF
  (a caller-supplied matching GENCODE/Ensembl GTF),
  genome FASTA (same dir, `Gencode_v49_GRCh38.primary_assembly.genome.fa`, `.fai` present),
  the curated manifest (for the transcript set, splits, and positive labels).
- **Per transcript:** build the exon model (chrom, strand, sorted exon intervals from GTF
  `exon` features), splice into mRNA coordinates, walk every position, emit a row:
  `transcript_id, chrom, gstart (plus-strand genomic A), strand, mrna_index, codon (3-mer),
  codon_class, split`.
- **Labels:** `label_tis = 1` iff the `(transcript_id, mrna_index)` matches a called TIS
  (`label_tis==1`) in the curated manifest; else 0. Also carry `was_curated_candidate`
  (bool) for cross-checking against Phase 1.
- **`row_idx`:** assigned `0..N-1` contiguous over the scan manifest (so `store.py`'s
  row-alignment assertion holds for the scan store).

**Coordinate correctness is the central risk** → enforced by a test: for every curated
candidate of an enumerated transcript, the enumerator's `gstart` at the candidate's
`mrna_index` must equal the manifest's `gstart`, and the enumerated 3-mer must equal the
manifest's `codon`. If GTF splicing/strand is wrong, this fails loudly.

### Reused unchanged (GPU + assembly)

- `extract.py` on the scan manifest → `data/scan_parts/*.npz` (one SLURM array per backend
  spec, same as `run_tis_extract.sh`). Same embedding keys as the curated store.
- `store.py` → `data/scan_store/` (row-aligned to the scan manifest).
- New thin SLURM wrapper `scripts/run_tis_scan.sh` mirroring `run_tis_extract.sh` but with
  `--manifest data/scan_manifest.parquet --out-dir data/scan_parts`.

### New: global-caller evaluation (`src/tisiago/scan_eval.py`, CPU)

- **Train + calibrate on the curated store** (reuse `caller.fit_calibrated_head`): the head
  trains on `train` near-cognate/AUG candidates and calibrates on `val` — exactly Phase 1.
  The trained head transfers because the scan store shares the embedding keys.
- **Apply to the scan store** test rows; compute:
  - **`recall_at_fp_budget`** over enumerated **AUG + near_cognate** test codons — now at
    *true imbalance* (all enumerated near-cognate/AUG negatives, not 3:1). Report recall @
    ≤1 FP/transcript + the threshold. Expect it to drop vs the curated 0.733 — that is the
    honest number.
  - **Non-cognate grounding:** mean P and FPR on **non_cognate** test codons (never trained
    on) → expect ≈ 0. Report mean, 95th percentile, and FPR at the caller threshold.
  - **Imbalance context:** print the realized negative:positive ratio per transcript vs the
    curated 3:1, so the drop is interpretable.

## Metrics / done when

- `scan_eval` prints: true-imbalance recall @ ≤1 FP/transcript (with threshold), the
  curated-3:1 recall for reference, and the non-cognate mean-P + FPR.
- The enumeration correctness test passes (enumerated gstart/codon == manifest for all
  curated candidates).
- Non-cognate mean-P is low (the grounding holds); if it is *not* ≈0, that is a real,
  reportable finding (the head leans on codon identity / context artifacts) — surface it,
  do not hide it.

## Risks

- **GTF parsing** (exon order, strand, frame). Mitigated by the coordinate-equality test
  against the curated manifest — a strong, automatic check.
- **Scan store size:** ~millions of positions × 5,632 fp16 ≈ tens of GB for test+val. If
  too large, restrict feature keys to AG16k + Evo2 blk28 (the headline set) rather than the
  full key matrix. Decide at extraction time; note it in the store `config.yaml`.
- **Positive join** must be exact on `(transcript_id, mrna_index)`; a mismatch silently
  zeroes recall. The correctness test covers this.
- The GPU scan is a SLURM job, not a subagent step — the implementation plan delivers the
  CPU-testable code (enumerator, eval) via TDD; the dense extraction is launched separately
  like the existing curated run.

## Out of scope (later phases)

Genome-wide enumeration (Phase 2 scale-up), autoresearch head sweep (Phase 3), HeLa
efficiency regression (Phase 4).
