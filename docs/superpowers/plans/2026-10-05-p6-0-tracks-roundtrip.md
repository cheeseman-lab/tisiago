# P6.0 — Ribo-seq Tracks and Ribo-TISH Round-Trip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build stranded P-site Ribo-seq tracks from the aTIS BAMs and prove, by a Ribo-TISH round trip, that the tracks reproduce Jimmy's TIS calls — the data foundation for fine-tuning AlphaGenome (P6.1+).

**Architecture:** Pure CPU code under `src/tisiago/seq2func/`. BAM reads → Ribo-TISH-identical P-site positions → 1-bp stranded bigWigs. The round trip converts tracks back into synthetic 2-bp-read BAMs, runs Jimmy's exact `ribotish predict`, and compares calls with his `*_TIS_predict_all.txt`. Labels are rebuilt by calling swissisoform's own `run_sample` (no port). QC adds replicate concordance and CHX frame periodicity.

**Tech Stack:** Python 3.11, pysam, pyBigWig, pandas, numpy, pytest; Ribo-TISH 0.2.7 (system install); swissisoform-v2 conda env for labels; Slurm.

**Spec:** `docs/superpowers/specs/2026-10-05-p6-seq2func-ribo-tracks-design.md` (§3, §4 P6.0, §6 step 2)

## Global Constraints

- Data root: `/lab/barcheese01/aTIS_data/` (`TISIAGO_ATIS_DATA` in `.env`); never write inside it.
- Conditions: `HeLa, K562, U2OS, RPE1_Async, RPE1_Que, RPE1_Sen`; 2 replicates; libraries `tis` (harringtonine RPF), `chx` (CHX RPF), `rna` (TIS-condition input). iPSC excluded.
- Reference: `reference/Gencode_v49_GRCh38.primary_assembly.genome.fa`, `reference/gencode.v49.primary_assembly.annotation.gtf` under the data root.
- Ribo-TISH: `/usr/local/bin/python3.8/ribotish` (0.2.7; `TISIAGO_RIBOTISH`). Jimmy's command, verbatim: `ribotish predict -t <TIS r1,r2> -b <CHX r1,r2> --tispara <..> --ribopara <..> -g <gtf> -f <fasta> -o <out> --minaalen 3 --alt --seq --aaseq --verbose`.
- Jimmy's filter: `TISPvalue <= 0.01`, `RiboPvalue <= 0.01`, `FisherQvalue <= 0.05`.
- Ribo-TISH read rules to mirror: skip secondary; skip `NH > 5`; skip `MAPQ < 1`; read length = `query_alignment_length`; offset from `.para.py`; reads with a 5′ mismatch use the `'m0'` table (empty ⇒ dropped).
- Outputs under `data/p6/` (gitignored). Temp files in the working directory, never `/tmp`.
- Envs: CPU code in conda `tisiago` (`uv pip install -e ".[dev,tracks]"`); labels step in conda `swissisoform-v2`.
- Round-trip pass bar: per-condition Jaccard ≥ 0.95 on filtered `(Tid, Start)` calls.
- ruff line length 100, Google docstrings; match existing module style (short module docstring, no section dividers).

## Review Focus

- Minus-strand reads: Ribo-TISH places the P-site at `reference_start + (L − offset)`, one base 3′ of the naive position — expect positions to match Ribo-TISH, not intuition (Task 2 pins it).
- Spliced reads whose P-site lands at an exon junction must map to the 5′ end of the downstream exon (bias=1) on `+` and the mirrored rule on `−` (Task 2).
- Reads with a 5′ mismatch and a `'m0': {}` para table must be dropped, not counted with the default offset (Task 2).
- Fractional predicted values (later phases) must round, not truncate, when converted to synthetic reads; a position-0 P-site cannot be represented and must be skipped, not crash (Task 4).
- Two conditions running the round trip in parallel must not overwrite each other's Ribo-TISH background file (Task 6 passes a per-condition `-e`).

---

### Task 1: Configuration and sample table

**Files:**
- Modify: `pyproject.toml` (add `tracks` extra)
- Modify: `.env.example`, local `.env` (add two keys)
- Create: `src/tisiago/seq2func/__init__.py`
- Create: `src/tisiago/seq2func/samples.py`
- Test: `tests/test_seq2func_samples.py`

**Interfaces:**
- Produces: `CONDITIONS: tuple[str, ...]`, `LIBRARIES: tuple[str, ...]`, `condition_name(cell_line: str, condition: str) -> str`, `parse_para(path) -> tuple[dict[int, int], dict[int, int] | None]`, `load_samples(atis_root) -> pd.DataFrame` with columns `condition, library, rep, sample_id, bam, para` (36 rows; `para` is `None` for `rna`).

- [ ] **Step 1: Add the extra and env keys**

In `pyproject.toml` under `[project.optional-dependencies]` add:

```toml
tracks = [
    "pysam>=0.22",
    "pyBigWig>=0.3.22",
]
```

Append to `.env.example`:

```
# P6 Ribo-seq source data and the Ribo-TISH executable used for the labels.
# TISIAGO_ATIS_DATA=/absolute/path/to/aTIS_data
# TISIAGO_RIBOTISH=/absolute/path/to/ribotish
```

Append to the local `.env`:

```
TISIAGO_ATIS_DATA=/lab/barcheese01/aTIS_data
TISIAGO_RIBOTISH=/usr/local/bin/python3.8/ribotish
```

Install: `eval "$(conda shell.bash hook)" && conda activate tisiago && uv pip install -e ".[dev,tracks]"`

- [ ] **Step 2: Write the failing tests**

```python
"""Tests for the aTIS sample table and Ribo-TISH offset files."""

import pandas as pd
import pytest

from tisiago.seq2func.samples import CONDITIONS, condition_name, load_samples, parse_para


def test_condition_name_maps_rpe1_states():
    assert condition_name("RPE1", "Quiescent") == "RPE1_Que"
    assert condition_name("RPE1", "Senescent") == "RPE1_Sen"
    assert condition_name("RPE1", "Async") == "RPE1_Async"
    assert condition_name("HeLa", "") == "HeLa"


def test_parse_para_splits_m0_table(tmp_path):
    path = tmp_path / "x.bam.para.py"
    path.write_text("offdict = {32: 13, 33: 13, 'm0': {}}\n")
    offsets, m0 = parse_para(path)
    assert offsets == {32: 13, 33: 13}
    assert m0 == {}


def test_parse_para_without_m0(tmp_path):
    path = tmp_path / "x.bam.para.py"
    path.write_text("offdict = {2: 1}\n")
    assert parse_para(path) == ({2: 1}, None)


def _sheet(tmp_path):
    rows = []
    lines = [("HeLa", ""), ("K562", ""), ("U2OS", ""), ("RPE1", "Async"),
             ("RPE1", "Quiescent"), ("RPE1", "Senescent")]
    for cell_line, condition in lines:
        for treatment in ("TIS", "CHX"):
            for kind in ("RPF", "Input"):
                for rep in (1, 2):
                    sid = f"{cell_line}{condition}_{treatment}_{kind}_{rep}"
                    rows.append({
                        "sample_name": sid, "cell_line": cell_line, "condition": condition,
                        "treatment": treatment, "rep": rep, "type": kind, "sample_id": sid,
                        "bam_path": f"ribosome_profiling/bam/{sid}.bam", "bedgraph_path": "",
                    })
    rows.append({"sample_name": "Harr_iPSC", "cell_line": "iPSC", "condition": "",
                 "treatment": "Harringtonine", "rep": "", "type": "RPF",
                 "sample_id": "HarrIPSC", "bam_path": "", "bedgraph_path": "x.bedgraph"})
    sheet_dir = tmp_path / "ribosome_profiling"
    sheet_dir.mkdir()
    pd.DataFrame(rows).to_csv(sheet_dir / "manifest.csv", index=False)
    return tmp_path


def test_load_samples_keeps_tis_chx_and_tis_input(tmp_path):
    samples = load_samples(_sheet(tmp_path))
    assert len(samples) == 36
    assert set(samples.condition) == set(CONDITIONS)
    assert set(samples.library) == {"tis", "chx", "rna"}
    rna = samples[samples.library == "rna"]
    assert rna.para.isna().all()
    assert all(str(p).endswith(".bam.para.py") for p in samples[samples.library != "rna"].para)


def test_load_samples_rejects_incomplete_sheet(tmp_path):
    root = _sheet(tmp_path)
    sheet = pd.read_csv(root / "ribosome_profiling" / "manifest.csv")
    sheet.iloc[1:].to_csv(root / "ribosome_profiling" / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="expected 36"):
        load_samples(root)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_seq2func_samples.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tisiago.seq2func'`

- [ ] **Step 4: Implement**

`src/tisiago/seq2func/__init__.py`:

```python
"""P6 seq2func: Ribo-seq tracks, Ribo-TISH round trip, and fine-tuning support."""
```

`src/tisiago/seq2func/samples.py`:

```python
"""The aTIS Ribo-seq sample sheet and Ribo-TISH P-site offset files."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

CONDITIONS = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")
LIBRARIES = ("tis", "chx", "rna")
REPLICATES = (1, 2)
_RPE1 = {"Async": "RPE1_Async", "Quiescent": "RPE1_Que", "Senescent": "RPE1_Sen"}
_LIBRARY = {("TIS", "RPF"): "tis", ("CHX", "RPF"): "chx", ("TIS", "Input"): "rna"}


def condition_name(cell_line: str, condition: str) -> str:
    """Return the condition label used by Ribo-TISH outputs and the tisiago manifest."""
    if cell_line == "RPE1":
        return _RPE1[condition]
    return cell_line


def parse_para(path: str | Path) -> tuple[dict[int, int], dict[int, int] | None]:
    """Parse a Ribo-TISH ``.para.py`` file into (offsets, m0 offsets or None)."""
    _, _, literal = Path(path).read_text().partition("=")
    offdict = ast.literal_eval(literal.strip())
    m0 = offdict.pop("m0", None)
    offsets = {int(length): int(offset) for length, offset in offdict.items()}
    if m0 is not None:
        m0 = {int(length): int(offset) for length, offset in m0.items()}
    return offsets, m0


def load_samples(atis_root: str | Path) -> pd.DataFrame:
    """Load the 6 conditions x 3 libraries x 2 replicates used by P6."""
    root = Path(atis_root)
    sheet = pd.read_csv(root / "ribosome_profiling" / "manifest.csv", dtype=str).fillna("")
    rows = []
    for record in sheet[sheet.bam_path != ""].itertuples():
        library = _LIBRARY.get((record.treatment, record.type))
        if library is None:
            continue
        bam = root / record.bam_path
        rows.append(
            {
                "condition": condition_name(record.cell_line, record.condition),
                "library": library,
                "rep": int(record.rep),
                "sample_id": record.sample_id,
                "bam": bam,
                "para": None if library == "rna" else bam.with_name(bam.name + ".para.py"),
            }
        )
    samples = pd.DataFrame(rows)
    expected = len(CONDITIONS) * len(LIBRARIES) * len(REPLICATES)
    if len(samples) != expected or samples.duplicated(["condition", "library", "rep"]).any():
        raise ValueError(f"expected {expected} unique condition/library/rep samples")
    return samples.sort_values(["condition", "library", "rep"]).reset_index(drop=True)
```

- [ ] **Step 5: Run tests to verify they pass, then check the real sheet**

Run: `python -m pytest tests/test_seq2func_samples.py -v` — Expected: 5 PASS.
Run: `python -c "from tisiago.seq2func.samples import load_samples; s=load_samples('/lab/barcheese01/aTIS_data'); print(s.groupby(['condition','library']).size()); assert all(p.exists() for p in s.bam)"`
Expected: 18 groups of 2; no assertion error.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example src/tisiago/seq2func/__init__.py src/tisiago/seq2func/samples.py tests/test_seq2func_samples.py
git commit -m "feat(p6): aTIS sample table and Ribo-TISH offset parsing"
```

---

### Task 2: Ribo-TISH-identical P-site positions

**Files:**
- Create: `src/tisiago/seq2func/psite.py`
- Test: `tests/test_seq2func_psite.py`

**Interfaces:**
- Consumes: offsets from `samples.parse_para`.
- Produces: `MAX_NH = 5`, `MIN_MAPQ = 1`, `ribotish_genome_pos(reference_start: int, reference_end: int, cigartuples, is_reverse: bool, cdna_length: int, p: int, bias: int = 1) -> int | None`, `is_m0(read) -> bool`, `psite(read, offsets: dict[int, int], m0_offsets: dict[int, int] | None) -> int | None`, `rna_end5(read) -> int | None`. All positions 0-based genomic.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for Ribo-TISH-identical P-site placement."""

import pysam

from tisiago.seq2func.psite import psite, ribotish_genome_pos, rna_end5

M, N = 0, 3
HEADER = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chr1", "LN": 10_000}]})


def _read(start, cigar, reverse=False, md=None, nh=1, mapq=255, secondary=False):
    read = pysam.AlignedSegment(HEADER)
    length = sum(n for op, n in cigar if op == M)
    read.query_name = "r"
    read.query_sequence = "A" * length
    read.reference_id = 0
    read.reference_start = start
    read.cigartuples = cigar
    read.mapping_quality = mapq
    read.flag = (16 if reverse else 0) | (256 if secondary else 0)
    read.set_tag("NH", nh)
    read.set_tag("MD", md or str(length))
    return read


def test_plus_strand_offset():
    assert ribotish_genome_pos(100, 130, [(M, 30)], False, 30, 12) == 112


def test_minus_strand_follows_ribotish_convention():
    assert ribotish_genome_pos(100, 130, [(M, 30)], True, 30, 12) == 118


def test_spliced_read_walks_blocks():
    cigar = [(M, 10), (N, 50), (M, 20)]
    assert ribotish_genome_pos(100, 180, cigar, False, 30, 12) == 162


def test_junction_maps_to_downstream_exon_start():
    cigar = [(M, 12), (N, 50), (M, 18)]
    assert ribotish_genome_pos(100, 180, cigar, False, 30, 12) == 162


def test_psite_applies_filters_and_lengths():
    offsets = {30: 12}
    assert psite(_read(100, [(M, 30)]), offsets, None) == 112
    assert psite(_read(100, [(M, 31)]), offsets, None) is None
    assert psite(_read(100, [(M, 30)], nh=6), offsets, None) is None
    assert psite(_read(100, [(M, 30)], mapq=0), offsets, None) is None
    assert psite(_read(100, [(M, 30)], secondary=True), offsets, None) is None


def test_psite_drops_five_prime_mismatch_with_empty_m0_table():
    plus_m0 = _read(100, [(M, 30)], md="0A29")
    minus_m0 = _read(100, [(M, 30)], reverse=True, md="29A0")
    assert psite(plus_m0, {30: 12}, {}) is None
    assert psite(minus_m0, {30: 12}, {}) is None
    assert psite(plus_m0, {30: 12}, None) == 112


def test_rna_end5_by_strand():
    assert rna_end5(_read(100, [(M, 30)])) == 100
    assert rna_end5(_read(100, [(M, 30)], reverse=True)) == 129
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_seq2func_psite.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
"""P-site placement that mirrors Ribo-TISH 0.2.7 (``zbio.bam`` / ``zbio.ribo``) exactly."""

from __future__ import annotations

MAX_NH = 5
MIN_MAPQ = 1
_MATCH = (0, 7, 8)
_INSERTION = 1
_SKIP = (2, 3)


def ribotish_genome_pos(
    reference_start: int,
    reference_end: int,
    cigartuples,
    is_reverse: bool,
    cdna_length: int,
    p: int,
    bias: int = 1,
) -> int | None:
    """Port of ``ribotish.zbio.bam.Read.genome_pos`` (single-end)."""
    if p < 0 or p > cdna_length:
        return None
    if p == 0:
        return reference_end if is_reverse else reference_start
    if p == cdna_length:
        return reference_start if is_reverse else reference_end
    p1 = p
    if is_reverse:
        p1 = cdna_length - p
        bias = 0 if bias == 1 else 1
    pos = reference_start
    for op, length in cigartuples:
        if op in _MATCH:
            if length - p1 >= bias:
                return pos + p1
            pos += length
            p1 -= length
        elif op == _INSERTION:
            if length - p1 >= bias:
                return pos
            p1 -= length
        elif op in _SKIP:
            pos += length
    return pos


def _passes_filters(read) -> bool:
    if read.is_unmapped or read.is_secondary or read.mapping_quality < MIN_MAPQ:
        return False
    return not (read.has_tag("NH") and read.get_tag("NH") > MAX_NH)


def is_m0(read) -> bool:
    """Ribo-TISH's 5'-end mismatch test on the MD tag."""
    md = read.get_tag("MD")
    if not read.is_reverse:
        return md[0] == "0"
    return md[-1] == "0" and not md[-2].isdigit()


def psite(read, offsets: dict[int, int], m0_offsets: dict[int, int] | None) -> int | None:
    """Genomic P-site of an RPF read, or None when Ribo-TISH would not count it."""
    if not _passes_filters(read):
        return None
    table = m0_offsets if m0_offsets is not None and is_m0(read) else offsets
    length = read.query_alignment_length
    offset = table.get(length)
    if offset is None:
        return None
    return ribotish_genome_pos(
        read.reference_start, read.reference_end, read.cigartuples, read.is_reverse, length, offset
    )


def rna_end5(read) -> int | None:
    """Genomic 5' end of an RNA-input read, with the same alignment filters."""
    if not _passes_filters(read) or (read.is_paired and not read.is_read1):
        return None
    return read.reference_end - 1 if read.is_reverse else read.reference_start
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_seq2func_psite.py -v` — Expected: 7 PASS.

- [ ] **Step 5: Cross-check against Ribo-TISH itself on real reads**

Ribo-TISH's own Python (`/usr/bin/python`) can import `ribotish.zbio.bam`. Write `tmp/psite_crosscheck.py` that, for the first 20,000 reads of `chr1` in `CGATGT_8_Aligned.sortedByCoord.out.bam`, prints `query_name, reference_start, Ribo-TISH genome_pos(offset(read, offdict))` using `ribotish.zbio.bam.Bam` / `ribotish.zbio.ribo.offset`, and compare with `psite()` from the tisiago env. Expected: identical positions for every read both tools count, and identical sets of dropped reads. Delete `tmp/psite_crosscheck.py` after.

- [ ] **Step 6: Commit**

```bash
git add src/tisiago/seq2func/psite.py tests/test_seq2func_psite.py
git commit -m "feat(p6): Ribo-TISH-identical P-site placement"
```

---

### Task 3: BAM → stranded 1-bp bigWig tracks

**Files:**
- Create: `src/tisiago/seq2func/tracks.py`
- Create: `scripts/run_p6_tracks.sh`
- Test: `tests/test_seq2func_tracks.py`

**Interfaces:**
- Consumes: `load_samples`, `parse_para`, `psite`, `rna_end5`.
- Produces: `count_positions(bam_path, position_fn) -> dict[tuple[str, str], collections.Counter]` (key `(chrom, "+"|"-")`), `write_bigwigs(counts, chrom_sizes: list[tuple[str, int]], prefix: Path) -> int` (writes `prefix.plus.bw`, `prefix.minus.bw`, returns total count), `read_bigwig_counts(path) -> dict[str, collections.Counter]`, `track_prefix(out_dir, condition, library, rep: int | str) -> Path` (`out_dir/condition/{library}_rep{rep}`; `rep="pooled"` for pooled). CLI: `python -m tisiago.seq2func.tracks --out-dir DIR (--index I | --pool)`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for stranded 1-bp bigWig track building."""

import pysam

from tisiago.seq2func.psite import rna_end5
from tisiago.seq2func.tracks import count_positions, read_bigwig_counts, write_bigwigs

SIZES = [("chr1", 1_000), ("chr2", 500)]


def _bam(tmp_path, reads):
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": c, "LN": n} for c, n in SIZES]}
    path = tmp_path / "x.bam"
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for i, (ref, start, reverse) in enumerate(reads):
            read = pysam.AlignedSegment(bam.header)
            read.query_name = f"r{i}"
            read.query_sequence = "A" * 10
            read.reference_id = ref
            read.reference_start = start
            read.cigartuples = [(0, 10)]
            read.mapping_quality = 255
            read.flag = 16 if reverse else 0
            read.set_tag("NH", 1)
            read.set_tag("MD", "10")
            bam.write(read)
    pysam.index(str(path))
    return path


def test_counts_roundtrip_through_bigwig(tmp_path):
    bam = _bam(tmp_path, [(0, 5, False), (0, 5, False), (0, 20, True), (1, 7, False)])
    counts = count_positions(bam, rna_end5)
    assert counts[("chr1", "+")][5] == 2
    assert counts[("chr1", "-")][29] == 1
    total = write_bigwigs(counts, SIZES, tmp_path / "t")
    assert total == 4
    plus = read_bigwig_counts(tmp_path / "t.plus.bw")
    minus = read_bigwig_counts(tmp_path / "t.minus.bw")
    assert plus["chr1"][5] == 2 and plus["chr2"][7] == 1
    assert minus["chr1"][29] == 1
    assert sum(minus["chr2"].values()) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_seq2func_tracks.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `tracks.py`**

```python
"""Stranded 1-bp Ribo-seq tracks: RPF P-sites (Ribo-TISH offsets) and RNA 5' ends."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pyBigWig
import pysam

from tisiago.seq2func.psite import psite, rna_end5
from tisiago.seq2func.samples import load_samples, parse_para

STRANDS = {"+": "plus", "-": "minus"}


def track_prefix(out_dir: str | Path, condition: str, library: str, rep: int | str) -> Path:
    """Path prefix of one track; ``rep='pooled'`` for the replicate sum."""
    return Path(out_dir) / condition / f"{library}_rep{rep}"


def count_positions(
    bam_path: str | Path, position_fn: Callable
) -> dict[tuple[str, str], Counter]:
    """Count reads per (chrom, strand) at the position returned by ``position_fn``."""
    counts: dict[tuple[str, str], Counter] = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            position = position_fn(read)
            if position is None:
                continue
            key = (read.reference_name, "-" if read.is_reverse else "+")
            counts.setdefault(key, Counter())[position] += 1
    return counts


def write_bigwigs(
    counts: dict[tuple[str, str], Counter], chrom_sizes: list[tuple[str, int]], prefix: Path
) -> int:
    """Write ``prefix.plus.bw`` / ``prefix.minus.bw`` at 1-bp resolution; return the total."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    for strand, name in STRANDS.items():
        bw = pyBigWig.open(str(prefix.with_name(f"{prefix.name}.{name}.bw")), "w")
        bw.addHeader(chrom_sizes)
        for chrom, size in chrom_sizes:
            counter = counts.get((chrom, strand))
            if not counter:
                continue
            positions = sorted(p for p in counter if 0 <= p < size)
            if not positions:
                continue
            values = [float(counter[p]) for p in positions]
            bw.addEntries(chrom, positions, values=values, span=1)
            total += int(sum(values))
        bw.close()
    return total


def read_bigwig_counts(path: str | Path) -> dict[str, Counter]:
    """Read a 1-bp bigWig back into per-chromosome position counters."""
    result: dict[str, Counter] = {}
    bw = pyBigWig.open(str(path))
    try:
        for chrom in bw.chroms():
            counter = Counter()
            for start, end, value in bw.intervals(chrom) or ():
                for position in range(start, end):
                    counter[position] += value
            result[chrom] = counter
    finally:
        bw.close()
    return result


def _chrom_sizes(bam_path: str | Path) -> list[tuple[str, int]]:
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return list(zip(bam.references, bam.lengths, strict=True))


def build_sample(sample, out_dir: Path) -> dict:
    """Build one sample's stranded track and write its depth record."""
    if sample.library == "rna":
        position_fn = rna_end5
    else:
        offsets, m0 = parse_para(sample.para)
        position_fn = lambda read: psite(read, offsets, m0)  # noqa: E731
    prefix = track_prefix(out_dir, sample.condition, sample.library, sample.rep)
    total = write_bigwigs(count_positions(sample.bam, position_fn), _chrom_sizes(sample.bam), prefix)
    record = {
        "condition": sample.condition,
        "library": sample.library,
        "rep": int(sample.rep),
        "sample_id": sample.sample_id,
        "total": total,
    }
    prefix.with_suffix(".json").write_text(json.dumps(record) + "\n")
    return record


def pool(out_dir: Path, condition: str, library: str, chrom_sizes) -> int:
    """Sum the two replicate tracks into ``{library}_reppooled``."""
    counts: dict[tuple[str, str], Counter] = {}
    for rep in (1, 2):
        prefix = track_prefix(out_dir, condition, library, rep)
        for strand, name in STRANDS.items():
            for chrom, counter in read_bigwig_counts(f"{prefix}.{name}.bw").items():
                counts.setdefault((chrom, strand), Counter()).update(counter)
    return write_bigwigs(counts, chrom_sizes, track_prefix(out_dir, condition, library, "pooled"))


def main() -> None:
    """CLI: build one sample (``--index``, a Slurm array task) or pool all replicates."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atis-root", default=os.environ.get("TISIAGO_ATIS_DATA"))
    ap.add_argument("--out-dir", default="data/p6/tracks")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--pool", action="store_true")
    args = ap.parse_args()
    samples = load_samples(args.atis_root)
    out_dir = Path(args.out_dir)
    if args.index is not None:
        print(json.dumps(build_sample(samples.iloc[args.index], out_dir)))
        return
    sizes = _chrom_sizes(samples.bam.iloc[0])
    for (condition, library), _ in samples.groupby(["condition", "library"]):
        print(condition, library, pool(out_dir, condition, library, sizes), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_seq2func_tracks.py -v` — Expected: PASS.

- [ ] **Step 5: Slurm wrapper** — `scripts/run_p6_tracks.sh` (36-task array, then pool):

```bash
#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6_tracks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Usage (from the repo root):
#   A=$(sbatch --parsable --partition=20 --array=0-35%12 scripts/run_p6_tracks.sh)
#   sbatch --partition=20 --dependency=afterok:$A scripts/run_p6_tracks.sh pool

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"

cd "$REPO_ROOT"
if [[ "${1:-}" == "pool" ]]; then
    "$PYTHON" -m tisiago.seq2func.tracks --out-dir data/p6/tracks --pool
else
    "$PYTHON" -m tisiago.seq2func.tracks --out-dir data/p6/tracks --index "$SLURM_ARRAY_TASK_ID"
fi
```

Run it. Expected: 36 `*.json` depth records plus 18 pooled track pairs under `data/p6/tracks/`.

- [ ] **Step 6: Commit**

```bash
git add src/tisiago/seq2func/tracks.py scripts/run_p6_tracks.sh tests/test_seq2func_tracks.py
git commit -m "feat(p6): stranded 1-bp P-site and RNA bigWig tracks from aTIS BAMs"
```

---

### Task 4: Tracks → synthetic Ribo-TISH BAMs

**Files:**
- Create: `src/tisiago/seq2func/synthetic_bam.py`
- Test: `tests/test_seq2func_synthetic_bam.py`

**Interfaces:**
- Consumes: `read_bigwig_counts`, `psite`, `parse_para`.
- Produces: `READ_LENGTH = 2`, `READ_OFFSET = 1`, `write_para(path) -> None` (writes `offdict = {2: 1}`), `tracks_to_bam(plus_bw, minus_bw, out_bam, scale: float = 1.0) -> int` (sorted, indexed BAM; returns read count). A 2-bp read starting at `x − 1` has its Ribo-TISH P-site at `x` on both strands.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for synthetic BAMs that reproduce a P-site track under Ribo-TISH rules."""

from collections import Counter

import pysam

from tisiago.seq2func.psite import psite
from tisiago.seq2func.samples import parse_para
from tisiago.seq2func.synthetic_bam import tracks_to_bam, write_para
from tisiago.seq2func.tracks import write_bigwigs

SIZES = [("chr1", 1_000)]


def _tracks(tmp_path, plus, minus):
    counts = {("chr1", "+"): Counter(plus), ("chr1", "-"): Counter(minus)}
    write_bigwigs(counts, SIZES, tmp_path / "t")
    return tmp_path / "t.plus.bw", tmp_path / "t.minus.bw"


def _psites(bam_path, para_path):
    offsets, m0 = parse_para(para_path)
    found = Counter()
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch("chr1"):
            found[("-" if read.is_reverse else "+", psite(read, offsets, m0))] += 1
    return found


def test_synthetic_bam_reproduces_track_under_ribotish_offsets(tmp_path):
    plus_bw, minus_bw = _tracks(tmp_path, {10: 3, 500: 1}, {20: 2})
    out = tmp_path / "s.bam"
    assert tracks_to_bam(plus_bw, minus_bw, out) == 6
    write_para(tmp_path / "s.bam.para.py")
    assert _psites(out, tmp_path / "s.bam.para.py") == Counter(
        {("+", 10): 3, ("+", 500): 1, ("-", 20): 2}
    )


def test_scale_rounds_and_position_zero_is_skipped(tmp_path):
    plus_bw, minus_bw = _tracks(tmp_path, {0: 4, 30: 1}, {})
    out = tmp_path / "s.bam"
    assert tracks_to_bam(plus_bw, minus_bw, out, scale=2.6) == 3  # round(2.6) at 30; 0 skipped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_seq2func_synthetic_bam.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
"""Synthetic BAMs whose Ribo-TISH P-sites reproduce a stranded 1-bp track.

Each count at position ``x`` becomes a 2-bp read starting at ``x - 1``. With the para table
``{2: 1}``, Ribo-TISH places the P-site at ``x`` on both strands, and a 2-bp read always passes
its splice-compatibility check (``--compatiblemis`` 2), so no transcript model is needed.
"""

from __future__ import annotations

from pathlib import Path

import pyBigWig
import pysam

from tisiago.seq2func.tracks import read_bigwig_counts

READ_LENGTH = 2
READ_OFFSET = 1


def write_para(path: str | Path) -> None:
    """Write the Ribo-TISH offset table that matches the synthetic reads."""
    Path(path).write_text(f"offdict = {{{READ_LENGTH}: {READ_OFFSET}}}\n")


def tracks_to_bam(
    plus_bw: str | Path, minus_bw: str | Path, out_bam: str | Path, scale: float = 1.0
) -> int:
    """Write a coordinate-sorted, indexed BAM for a stranded track; return the read count."""
    bw = pyBigWig.open(str(plus_bw))
    chrom_sizes = list(bw.chroms().items())
    bw.close()
    plus = read_bigwig_counts(plus_bw)
    minus = read_bigwig_counts(minus_bw)
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": c, "LN": n} for c, n in chrom_sizes]}
    n_reads = 0
    with pysam.AlignmentFile(str(out_bam), "wb", header=header) as bam:
        for reference_id, (chrom, _) in enumerate(chrom_sizes):
            events = []
            for is_reverse, counter in ((False, plus.get(chrom, {})), (True, minus.get(chrom, {}))):
                for position, value in counter.items():
                    count = int(round(value * scale))
                    if position >= READ_OFFSET and count > 0:
                        events.append((position - READ_OFFSET, is_reverse, count))
            for start, is_reverse, count in sorted(events):
                for _ in range(count):
                    read = pysam.AlignedSegment(bam.header)
                    read.query_name = f"s{n_reads}"
                    read.query_sequence = "N" * READ_LENGTH
                    read.reference_id = reference_id
                    read.reference_start = start
                    read.cigartuples = [(0, READ_LENGTH)]
                    read.mapping_quality = 255
                    read.flag = 16 if is_reverse else 0
                    read.set_tag("NH", 1)
                    read.set_tag("MD", str(READ_LENGTH))
                    bam.write(read)
                    n_reads += 1
    pysam.index(str(out_bam))
    return n_reads
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_seq2func_synthetic_bam.py -v` — Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tisiago/seq2func/synthetic_bam.py tests/test_seq2func_synthetic_bam.py
git commit -m "feat(p6): synthetic 2-bp-read BAMs that reproduce P-site tracks in Ribo-TISH"
```

---

### Task 5: Jimmy's Ribo-TISH command, filter and call comparison

**Files:**
- Create: `src/tisiago/seq2func/ribotish.py`
- Test: `tests/test_seq2func_ribotish.py`

**Interfaces:**
- Produces: `predict_command(ribotish: str, tis_bams: list, chx_bams: list, tis_paras: list, chx_paras: list, gtf, fasta, output, estimate) -> list[str]`, `filter_predictions(df: pd.DataFrame) -> pd.DataFrame`, `call_keys(df) -> set[tuple[str, int]]` (`(Tid, Start)`), `compare_calls(observed: pd.DataFrame, reference: pd.DataFrame) -> dict` with keys `n_observed, n_reference, n_shared, jaccard`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the Ribo-TISH command, Jimmy's filter, and call comparison."""

import pandas as pd

from tisiago.seq2func.ribotish import (
    call_keys,
    compare_calls,
    filter_predictions,
    predict_command,
)


def test_predict_command_matches_jimmys_flags():
    argv = predict_command(
        "ribotish", ["t1.bam", "t2.bam"], ["c1.bam", "c2.bam"], ["t1.p", "t2.p"],
        ["c1.p", "c2.p"], "g.gtf", "g.fa", "out.txt", "bg.txt",
    )
    assert argv == [
        "ribotish", "predict", "-t", "t1.bam,t2.bam", "-b", "c1.bam,c2.bam",
        "--tispara", "t1.p,t2.p", "--ribopara", "c1.p,c2.p", "-g", "g.gtf", "-f", "g.fa",
        "-o", "out.txt", "-e", "bg.txt", "--minaalen", "3", "--alt", "--seq", "--aaseq",
        "--verbose",
    ]


def _calls(rows):
    return pd.DataFrame(rows, columns=["Tid", "Start", "TISPvalue", "RiboPvalue", "FisherQvalue"])


def test_filter_predictions_uses_jimmys_thresholds():
    df = _calls([("a", 1, 0.01, 0.01, 0.05), ("b", 2, 0.02, 0.0, 0.0),
                 ("c", 3, 0.0, 0.02, 0.0), ("d", 4, 0.0, 0.0, 0.06)])
    assert list(filter_predictions(df).Tid) == ["a"]


def test_compare_calls_jaccard():
    observed = _calls([("a", 1, 0, 0, 0), ("b", 2, 0, 0, 0)])
    reference = _calls([("a", 1, 0, 0, 0), ("c", 3, 0, 0, 0)])
    assert call_keys(observed) == {("a", 1), ("b", 2)}
    assert compare_calls(observed, reference) == {
        "n_observed": 2, "n_reference": 2, "n_shared": 1, "jaccard": 1 / 3,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_seq2func_ribotish.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
"""Jimmy's Ribo-TISH invocation and filter (aTIS_data/scripts/processing.sh), and call overlap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TIS_P_MAX = 0.01
RIBO_P_MAX = 0.01
FISHER_Q_MAX = 0.05


def predict_command(
    ribotish: str,
    tis_bams: list,
    chx_bams: list,
    tis_paras: list,
    chx_paras: list,
    gtf: str | Path,
    fasta: str | Path,
    output: str | Path,
    estimate: str | Path,
) -> list[str]:
    """Jimmy's pooled ``ribotish predict``; ``estimate`` isolates the background file."""
    join = lambda paths: ",".join(str(p) for p in paths)  # noqa: E731
    return [
        str(ribotish), "predict",
        "-t", join(tis_bams), "-b", join(chx_bams),
        "--tispara", join(tis_paras), "--ribopara", join(chx_paras),
        "-g", str(gtf), "-f", str(fasta), "-o", str(output), "-e", str(estimate),
        "--minaalen", "3", "--alt", "--seq", "--aaseq", "--verbose",
    ]


def filter_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Jimmy's significance filter on a ``*_TIS_predict_all.txt`` table."""
    keep = (
        (df.TISPvalue <= TIS_P_MAX)
        & (df.RiboPvalue <= RIBO_P_MAX)
        & (df.FisherQvalue <= FISHER_Q_MAX)
    )
    return df[keep]


def call_keys(df: pd.DataFrame) -> set[tuple[str, int]]:
    """Identify a call by transcript and transcript-coordinate start."""
    return set(zip(df.Tid.astype(str), df.Start.astype(int), strict=True))


def compare_calls(observed: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Overlap of two filtered call tables."""
    a, b = call_keys(observed), call_keys(reference)
    union = a | b
    return {
        "n_observed": len(a),
        "n_reference": len(b),
        "n_shared": len(a & b),
        "jaccard": len(a & b) / len(union) if union else 1.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_seq2func_ribotish.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tisiago/seq2func/ribotish.py tests/test_seq2func_ribotish.py
git commit -m "feat(p6): Jimmy's Ribo-TISH command, filter, and call comparison"
```

---

### Task 6: Round-trip contract on observed tracks

**Files:**
- Create: `src/tisiago/seq2func/roundtrip.py`
- Create: `scripts/run_p6_roundtrip.sh`
- Test: `tests/test_seq2func_roundtrip.py`

**Interfaces:**
- Consumes: `load_samples`, `track_prefix`, `tracks_to_bam`, `write_para`, `predict_command`, `filter_predictions`, `compare_calls`.
- Produces: `prepare_inputs(condition, tracks_dir, work_dir) -> dict[str, list[Path]]` (keys `tis_bams, chx_bams, tis_paras, chx_paras`), `run_condition(condition, tracks_dir, work_dir, atis_root, ribotish) -> dict` (comparison + condition), CLI `python -m tisiago.seq2func.roundtrip --condition C` writing `data/p6/roundtrip/{C}/report.json`; `PASS_JACCARD = 0.95`.

- [ ] **Step 1: Write the failing test (input preparation only; the Ribo-TISH run is exercised by the Slurm step)**

```python
"""Tests for round-trip input preparation."""

from collections import Counter

from tisiago.seq2func.roundtrip import prepare_inputs
from tisiago.seq2func.tracks import track_prefix, write_bigwigs


def test_prepare_inputs_builds_rep_bams_and_paras(tmp_path):
    tracks = tmp_path / "tracks"
    for library in ("tis", "chx"):
        for rep in (1, 2):
            counts = {("chr1", "+"): Counter({10: rep}), ("chr1", "-"): Counter()}
            write_bigwigs(counts, [("chr1", 100)], track_prefix(tracks, "HeLa", library, rep))
    inputs = prepare_inputs("HeLa", tracks, tmp_path / "work")
    assert [p.name for p in inputs["tis_bams"]] == ["tis_rep1.bam", "tis_rep2.bam"]
    assert [p.name for p in inputs["chx_paras"]] == ["chx_rep1.bam.para.py", "chx_rep2.bam.para.py"]
    assert all(p.exists() for paths in inputs.values() for p in paths)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_seq2func_roundtrip.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

```python
"""Round-trip contract: observed tracks -> synthetic BAMs -> Jimmy's Ribo-TISH -> his calls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from tisiago.seq2func.ribotish import compare_calls, filter_predictions, predict_command
from tisiago.seq2func.synthetic_bam import tracks_to_bam, write_para
from tisiago.seq2func.tracks import track_prefix

PASS_JACCARD = 0.95


def prepare_inputs(condition: str, tracks_dir: Path, work_dir: Path) -> dict[str, list[Path]]:
    """Synthetic per-replicate TIS and CHX BAMs plus matching para files."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, list[Path]] = {}
    for library in ("tis", "chx"):
        bams, paras = [], []
        for rep in (1, 2):
            prefix = track_prefix(tracks_dir, condition, library, rep)
            bam = work_dir / f"{library}_rep{rep}.bam"
            tracks_to_bam(f"{prefix}.plus.bw", f"{prefix}.minus.bw", bam)
            para = bam.with_name(bam.name + ".para.py")
            write_para(para)
            bams.append(bam)
            paras.append(para)
        inputs[f"{library}_bams"] = bams
        inputs[f"{library}_paras"] = paras
    return inputs


def run_condition(
    condition: str, tracks_dir: Path, work_dir: Path, atis_root: Path, ribotish: str
) -> dict:
    """Run Jimmy's Ribo-TISH on synthetic BAMs and compare with his pooled calls."""
    work_dir, atis_root = Path(work_dir), Path(atis_root)
    inputs = prepare_inputs(condition, tracks_dir, work_dir)
    output = work_dir / f"{condition}_TIS_predict.txt"
    argv = predict_command(
        ribotish, inputs["tis_bams"], inputs["chx_bams"], inputs["tis_paras"],
        inputs["chx_paras"],
        atis_root / "reference" / "gencode.v49.primary_assembly.annotation.gtf",
        atis_root / "reference" / "Gencode_v49_GRCh38.primary_assembly.genome.fa",
        output, work_dir / "tisBackground.txt",
    )
    subprocess.run(argv, check=True, cwd=work_dir)
    ours = pd.read_csv(work_dir / f"{condition}_TIS_predict_all.txt", sep="\t")
    jimmy = pd.read_csv(
        atis_root / "ribotish" / "combined" / f"{condition}_TIS_predict_all.txt", sep="\t"
    )
    report = {"condition": condition, **compare_calls(filter_predictions(ours),
                                                      filter_predictions(jimmy))}
    report["passed"] = report["jaccard"] >= PASS_JACCARD
    return report


def main() -> None:
    """CLI: one condition per call (a Slurm array task)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--tracks-dir", default="data/p6/tracks")
    ap.add_argument("--out-dir", default="data/p6/roundtrip")
    ap.add_argument("--atis-root", default=os.environ.get("TISIAGO_ATIS_DATA"))
    ap.add_argument("--ribotish", default=os.environ.get("TISIAGO_RIBOTISH", "ribotish"))
    args = ap.parse_args()
    work_dir = Path(args.out_dir) / args.condition
    report = run_condition(
        args.condition, Path(args.tracks_dir), work_dir, Path(args.atis_root), args.ribotish
    )
    (work_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_seq2func_roundtrip.py -v` — Expected: PASS.

- [ ] **Step 5: Slurm wrapper** — `scripts/run_p6_roundtrip.sh` (6-task array; resources match Jimmy's job: 8 CPUs, 64 GB):

```bash
#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_p6_roundtrip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# Usage: sbatch --partition=20 --array=0-5 scripts/run_p6_roundtrip.sh

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
CONDITIONS=(HeLa K562 U2OS RPE1_Async RPE1_Que RPE1_Sen)
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.seq2func.roundtrip --condition "${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
```

- [ ] **Step 6: Run the contract and record the result**

Run the array after Task 3's tracks exist. Expected: six `report.json` files with `"passed": true` (Jaccard ≥ 0.95). If a condition fails, do not tune thresholds: compare `n_observed` vs `n_reference`, then diff P-site counts for the disagreeing transcripts between our track and Ribo-TISH's `--transprofile` output on the real BAMs (run `ribotish predict ... --transprofile prof.txt` on one condition) to locate the discrepancy. Known expected source of small differences: synthetic reads are always splice-compatible, whereas real reads that contradict a transcript's junctions are excluded for that transcript.

- [ ] **Step 7: Commit**

```bash
git add src/tisiago/seq2func/roundtrip.py scripts/run_p6_roundtrip.sh tests/test_seq2func_roundtrip.py
git commit -m "feat(p6): Ribo-TISH round-trip contract on observed tracks"
```

---

### Task 7: Per-condition labels via swissisoform, and track QC

**Files:**
- Create: `scripts/p6_build_labels.py` (runs in conda `swissisoform-v2`)
- Create: `src/tisiago/seq2func/qc.py`
- Test: `tests/test_seq2func_qc.py`

**Interfaces:**
- Consumes: round-trip `{condition}_TIS_predict_all.txt` files; RNA count files from `/lab/barcheese01/mdiberna/swissisoform-v2/data/reference/ribotish_replicate_manifest.csv`.
- Produces: `data/p6/labels/{condition}_labels.parquet` (swissisoform `run_sample` final table; positives are rows with `Imputed == False`); `data/p6/expression/{condition}.parquet` (columns `Gid`, `GeneRNASeqCounts`, `expressed`); `qc.replicate_concordance(rep1: dict, rep2: dict, regions: list[tuple[str, int, int]]) -> float` (Pearson of `log1p` region totals); `qc.frame_fraction(counts: dict[str, Counter], cds: list[tuple[str, str, list[tuple[int, int]]]]) -> float` (fraction of CDS P-sites in frame 0); `data/p6/qc.tsv`.

- [ ] **Step 1: Write the failing QC tests**

```python
"""Tests for replicate concordance and CHX frame periodicity."""

from collections import Counter

import pytest

from tisiago.seq2func.qc import frame_fraction, replicate_concordance


def test_replicate_concordance_is_one_for_proportional_tracks():
    rep1 = {"chr1": Counter({5: 1, 50: 10, 90: 100})}
    rep2 = {"chr1": Counter({5: 1, 50: 10, 90: 100})}
    regions = [("chr1", 0, 10), ("chr1", 40, 60), ("chr1", 80, 100)]
    assert replicate_concordance(rep1, rep2, regions) == pytest.approx(1.0)


def test_frame_fraction_counts_codon_first_positions_across_exons():
    # + strand CDS over exons [10, 16) and [30, 36): codon positions 10,13,30,33 are frame 0.
    cds = [("chr1", "+", [(10, 16), (30, 36)])]
    counts = {"chr1": Counter({10: 4, 11: 1, 30: 3, 34: 2})}
    assert frame_fraction(counts, cds) == pytest.approx(7 / 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_seq2func_qc.py -v` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `qc.py`**

```python
"""Track QC: replicate concordance (the accuracy ceiling) and CHX 3-nt periodicity."""

from __future__ import annotations

from collections import Counter

import numpy as np


def replicate_concordance(
    rep1: dict[str, Counter], rep2: dict[str, Counter], regions: list[tuple[str, int, int]]
) -> float:
    """Pearson correlation of log1p region totals between two replicate tracks."""
    def totals(track):
        return np.array(
            [sum(v for p, v in track.get(c, {}).items() if s <= p < e) for c, s, e in regions],
            dtype=np.float64,
        )

    a, b = np.log1p(totals(rep1)), np.log1p(totals(rep2))
    return float(np.corrcoef(a, b)[0, 1])


def frame_fraction(
    counts: dict[str, Counter], cds: list[tuple[str, str, list[tuple[int, int]]]]
) -> float:
    """Fraction of CDS P-site counts on the first base of a codon (same-strand track)."""
    in_frame = total = 0
    for chrom, strand, exons in cds:
        track = counts.get(chrom, {})
        positions = [p for start, end in sorted(exons) for p in range(start, end)]
        if strand == "-":
            positions = positions[::-1]
        for index, position in enumerate(positions):
            value = track.get(position, 0)
            total += value
            if index % 3 == 0:
                in_frame += value
    return in_frame / total if total else float("nan")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_seq2func_qc.py -v` — Expected: 2 PASS.

- [ ] **Step 5: Labels script** — `scripts/p6_build_labels.py`, run with the swissisoform-v2 env so swissisoform's own code builds the labels:

```python
"""Rebuild per-condition TIS labels from round-trip Ribo-TISH output with swissisoform."""

import argparse
from pathlib import Path

import pandas as pd
from swissisoform.io.rnaseq import sum_replicate_counts
from swissisoform.pipeline import run_sample

SWISS = Path("/lab/barcheese01/mdiberna/swissisoform-v2")
CONDITIONS = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")


def main() -> None:
    """Run swissisoform ``run_sample`` on each condition's round-trip predictions."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roundtrip-dir", default="data/p6/roundtrip")
    ap.add_argument("--atis-root", default="/lab/barcheese01/aTIS_data")
    ap.add_argument("--out-dir", default="data/p6/labels")
    ap.add_argument("--min-gene-counts", type=float, default=10.0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = pd.read_csv(SWISS / "data/reference/ribotish_replicate_manifest.csv")
    gtf = Path(args.atis_root) / "reference/gencode.v49.primary_assembly.annotation.gtf"
    for condition in CONDITIONS:
        counts = [SWISS / p for p in reps[reps["sample"] == condition].rnaseq_count_file]
        predict = Path(args.roundtrip_dir) / condition / f"{condition}_TIS_predict_all.txt"
        final, _ = run_sample(predict, counts, gtf, sample=condition)
        final.to_parquet(out_dir / f"{condition}_labels.parquet", index=False)
        expression = sum_replicate_counts(counts).rename_axis("Gid").reset_index()
        expression["expressed"] = expression.GeneRNASeqCounts >= args.min_gene_counts
        expression_dir = out_dir.parent / "expression"
        expression_dir.mkdir(parents=True, exist_ok=True)
        expression.to_parquet(expression_dir / f"{condition}.parquet", index=False)
        print(condition, len(final), int((~final.Imputed).sum()),
              int(expression.expressed.sum()), flush=True)


if __name__ == "__main__":
    main()
```

Run: `eval "$(conda shell.bash hook)" && conda activate swissisoform-v2 && python scripts/p6_build_labels.py`. If `run_sample` needs `genome_fasta`/`protein_fasta` for imputation, pass `--genome`/`--protein` paths from `swissisoform-v2/data/reference/` (`Gencode_v49_GRCh38.primary_assembly.genome.fa`, `gencode.v49.pc_translations.fa`) through to `run_sample(..., genome_fasta=..., protein_fasta=...)`. Then compare per-condition positives with the tisiago manifest's `present_{condition}` flags, and the `expressed` column with the manifest's `expressed_{condition}` flags (choose `--min-gene-counts` as the smallest value with ≥ 0.99 agreement and rerun if the default misses), and record both agreements in `data/p6/qc.tsv`. The `expressed` column is the loss mask for P6.2.

- [ ] **Step 6: QC table** — add a `main()` to `qc.py` that, for each condition × library, computes `replicate_concordance` over GENCODE v49 MANE transcripts' genomic spans (both strands summed) and, for `chx` pooled tracks, `frame_fraction` over MANE CDS exons; writes `data/p6/qc.tsv` with columns `condition, library, replicate_r, frame_fraction`. Run it on a CPU node (`--mem=64G`).

- [ ] **Step 7: Commit**

```bash
git add scripts/p6_build_labels.py src/tisiago/seq2func/qc.py tests/test_seq2func_qc.py
git commit -m "feat(p6): swissisoform-built per-condition labels and track QC"
```

---

### Task 8: Record P6.0 results

**Files:**
- Modify: `ROADMAP.md` (P6 row + P6 section), `FINDINGS.md` (new §11 "P6.0 tracks and round trip")

- [ ] **Step 1:** Add the per-condition round-trip table (n_observed, n_reference, Jaccard, passed), label counts vs `present_*` agreement, replicate concordance and CHX frame fraction to FINDINGS §11; set ROADMAP P6 status to "P6.0 done" with the pass/fail outcome.
- [ ] **Step 2: Run the full suite** — `python -m pytest -q` and `ruff check src tests scripts`; Expected: all pass.
- [ ] **Step 3: Commit**

```bash
git add ROADMAP.md FINDINGS.md
git commit -m "docs(p6): P6.0 tracks, Ribo-TISH round trip, labels and QC results"
```

---

## Later plans (written after P6.0 is reviewed)

Each is a separate plan because it depends on P6.0's outputs and on inspecting `alphagenome-ft` in its own environment:

- **P6.0b** approach note (no code): multi-cell-type practice in Borzoi / AlphaGenome / Enformer / Koo-lab work.
- **P6.1** `.venv/agft` + custom 1-bp head + one-region overfit (spec §4 P6.1).
- **P6.2** shared (36-output) and per-condition (6 × 6) heads-only arms with expression masking and Poisson-multinomial loss.
- **P6.3** conditional LoRA / partial unfreeze.
- **P6.4** predicted-track calls through Tasks 4–6 of this plan (scale = real library depth), §7 decision, single test-chromosome scoring vs `data/calls/ag_w8k_test_calls.parquet`.
