# Data contract — what the label pipeline hands to tisiago

What tisiago needs from the label pipeline (the ribotish fork, `0.2.8+cheeseman.1`, release `results/<run>/release/`
and `results/<run>/tracks/`) to run re-scope spec Phase 0-3. Code that reads it: `src/tisiago/truth/` (truth table,
P-site check, differential starts) and `src/tisiago/evaluation/start_metrics.py` (M1, M2). The column maps in
`truth/load.py` are the only place the format is spelled out; edit them, not the code, when the format changes.

## 1. Files per condition

Conditions: K562, HeLa, U2OS, RPE1_Async, RPE1_Que, RPE1_Sen. RPE1's three states are one leave-one-line-out group.

- **Ribo-seq P-site tracks, 1 bp, stranded.** Harringtonine (TIS) and CHX, **pooled and each replicate** (rep1, rep2):
  `<C>_{TIS,CHX}[_rep<N>]_{plus,minus}.bw`. Pooled is the training target; replicates define truth (spec §6).
  Each track's read total (`<C>.tracks.json` `total_psites`), so counts can be recovered from CPM.
- **Matched RNA-input, same lysate**, stranded, per replicate if it exists. What the model consumes is the RNA
  5'-end count track (`seq2func/psite.py` `rna_end5`), not coverage.
- **Truth table** (§2), all conditions in one file.
- **Transcript annotation** the calls were made on: exon blocks and genomic CDS per transcript (release `transcripts`).
- **Per-library P-site offsets** (`para` files) and BAM paths, so tracks can be rebuilt here if needed.

## 2. Truth table

One row per condition x start (pooled call). Source columns as in release v50 `starts`:

| tisiago column | v50 source | Notes |
|---|---|---|
| `condition` | `line` | |
| `transcript_id` | `Tid` | the transcript chosen for the start |
| `gene` | `Symbol` | |
| `chrom`, `strand` | `Chromosome`, `Strand` | `chr`-prefixed; splits are keyed on `chr7`/`chr8`/`chr9` |
| `gpos` | `Locus` | converted: `gpos = Locus` on +, `Locus - 1` on - (codon's first base, 0-based) |
| `tx_pos` | `Start` | asserted equal to the annotation's mapping of `gpos` |
| `codon` | `StartCodon` | DNA 3-mer |
| `rep1`, `rep2` | `rep1_called`, `rep2_called` | nullable bool; NA = no replicate (iPSC, CM) |
| — | `Imputed` | `True` rows are dropped: canonical starts added without a call |
| `pause` | none in v50 | nullable bool; NA until a source exists |

Computed on load, never read from the input: `codon_class` (AUG / AAG / AGG / other_near_cognate / non_cognate),
`region` (utr5 / annotated / cds_inframe / cds_outframe / other, from the transcript's own CDS and frame),
`reproducible` (rep1 and rep2), `line` (leave-one-out group), `split` (val chr7, test chr8/9, else train).
`load_truth` rejects missing columns, bad codons or strands, non-exonic starts, a `tx_pos` that disagrees with the
annotation, and duplicate (condition, chrom, strand, gpos). `summarize_truth` gives the Phase 0 count table.

## 3. Coordinates

- `gpos`: 0-based genomic base of the start codon's first transcribed base. + codon `gpos..gpos+2`; - `gpos-2..gpos`.
- Ribo-TISH `Locus` and the baseline manifest's `gstart` are both `gpos` on + and `gpos + 1` on - (half-open end).
- Transcript positions are 0-based, 5' -> 3' on the spliced transcript.
- CDS bounds are genomic, lowest coordinate, half-open, start and stop codons included.

## 4. P-site convention to assert

Harringtonine peaks at a start codon's first base. A track in the right convention peaks at offset 0 on **both**
strands around annotated AUG starts.

- tisiago tracks (`seq2func/psite.py`) mirror Ribo-TISH 0.2.7 and keep the minus-strand P-site at canonical + 1
  (`RIBOTISH_MINUS_SHIFT`). The fork's `make_tracks.psite_coord` subtracts it. So the two differ by 1 nt on -.
- Check: `truth.psite_check.psite_offsets(bigwig_reader(...), annotated AUG starts)` then `check_psite_convention`
  on each artifact, and `check_same_convention(model, release)`. A minus-strand offset one below plus is the 0.2.7
  shift. Run it on the first release bigWig before anything else; record the offsets.
- Ribo-TISH's own calls (`Start`, `TISCounts`) are internally consistent on both strands; only genome tracks differ.

## 5. Splits

Frozen: val chr7, test chr8/chr9, train elsewhere. Leave-one-line-out over K562 / HeLa / U2OS / RPE1. Test is scored
once per model behind `--i-am-final`. The release carries no split column; tisiago derives it from `chrom`.

## 6. Unknowns to settle on hand-off

1. **Which table is truth.** Raw `starts` with replicate flags, or the cross-line-filtered strong-start set
   (P6_PROJECT_PLAN §2)? If filtered, does it keep per-line `rep1`/`rep2`, and are dropped starts kept somewhere as
   known negatives or as "not evaluated"?
2. **Replicate support semantics.** `rep{1,2}_called` = the replicate's clean table calls the site, i.e. passed the
   full cascade in that replicate. Is that the intended bar, or is read support (TIS counts above background) wanted?
3. **Pause flag source.** No `chx_pause_*` column in v50. Computed here from CHX replicate tracks, or supplied?
4. **Region reference CDS.** `region` uses the row transcript's own CDS. The release's `RecatTISType` uses the gene's
   canonical CDS. Which one defines in-CDS for M1 (FINDINGS §17 used MANE)?
5. **P-site convention of the new bigWigs.** Confirm `make_tracks` is unchanged (fork convention, minus corrected),
   and whether upstream 0.2.7 -> 0.2.8 changed placement itself.
6. **Track units.** Release bigWigs are CPM. Confirm `total_psites` is the CPM denominator per file, or ship counts.
7. **RNA input.** Release RNA is pooled full-read coverage (CPM). The model uses per-library 5'-end counts. Ship
   per-replicate RNA BAMs (or 5'-end tracks), and confirm which RNA libraries pair with which Ribo-seq replicate.
8. **Chromosome naming and genome build** (`chr7` vs `7`; GRCh38 primary assembly) and GENCODE release (v50) of
   the annotation, which must match the one the model's transcript cache was built on.
9. **Conditions in scope.** iPSC and CM (Chen et al.) have no replicates, so no reproducible truth; confirm they are
   excluded from training and from M1/M2.
10. **Differential starts.** M2 needs per-replicate start counts per gene. Derived here from the replicate TIS tracks
    at truth sites, or supplied? Which grouping defines a share (gene vs transcript)?
