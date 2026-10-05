# Phase 2: Global All-Codon Caller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GTF-driven codon enumerator (`enumerate_codons.py`) that produces a dense **scan manifest** of every codon in held-out transcripts, a SLURM wrapper to embed it through the existing `extract.py`, and `scan_eval.py` that applies the Phase-1 calibrated head to the dense scan store and reports recall at true imbalance + the non-cognate≈0 grounding.

**Architecture:** Reuse the existing pipeline. The enumerator emits a parquet schema-compatible with `extract.py` (columns `row_idx, transcript_id, chrom, gstart, strand, codon`), so `extract.py` + `store.py` run unchanged on it. The enumerator's correctness is pinned by a test: for every existing curated candidate, the enumerated `(gstart, codon)` must equal the curated manifest's — catching any 1-based/0-based or strand error. `scan_eval.py` trains+calibrates on the curated store (Phase 1's `caller.fit_calibrated_head`) and applies the head to the scan store.

**Tech Stack:** numpy, pandas, pyfaidx (genome fetch — in the `[extract]` extra), manual GTF parsing, scikit-learn (via `caller.py`), pytest. CPU only. The dense GPU extraction is launched separately via SLURM after this code lands.

**Environment for every command:**
```bash
eval "$(conda shell.bash hook)" && conda activate tisiago
```
The enumerator needs `pyfaidx` (Task 4 installs it into the env). Reference files:
- GTF: `/lab/barcheese01/mdiberna/swissisoform-v2/data/reference/gencode.v49.primary_assembly.annotation.gtf`
- Genome: `/lab/barcheese01/mdiberna/swissisoform-v2/data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa`

**Coordinate conventions (verified, critical):**
- GTF coords are **1-based inclusive**; manifest `gstart` is **0-based**.
- `+` strand: a transcript's exons concatenated 5'→3' in ascending genomic order; `mrna_index=i` → genomic 0-based position of the codon's first base; `codon` is the genome 3-mer there.
- `-` strand: exons concatenated in **descending** genomic order, sequence reverse-complemented; the manifest stores `gstart` such that `tiling.a_plus_of(gstart, strand)` = the plus-strand position of the codon's A (for `-`, `a_plus = gstart - 1`). Match the manifest empirically via the Task-4 correctness test.

---

## File Structure

- **Create `src/tisiago/enumerate_codons.py`** — pure coordinate/codon functions + GTF parsing + a `main()` that writes `data/scan_manifest.parquet`. One responsibility: turn the GTF + genome + curated manifest into a dense, labelled, schema-compatible scan manifest.
- **Create `tests/test_enumerate_codons.py`** — unit tests for the pure functions on synthetic exons/GTF, plus one integration test against the real curated manifest.
- **Create `src/tisiago/scan_eval.py`** — apply the curated-trained calibrated head to the scan store; report true-imbalance recall + non-cognate grounding.
- **Create `tests/test_scan_eval.py`** — unit test the metric-combining logic on synthetic arrays.
- **Create `scripts/run_tis_scan.sh`** — SLURM wrapper mirroring `run_tis_extract.sh`, pointed at the scan manifest (written, not executed here).
- **Modify `ARCHITECTURE.md`, `CLAUDE.md`** — add the scan modules + `scan_store/` layout.

---

## Task 1: Codon classification (pure)

`classify_codon` maps a 3-mer to `{"AUG", "near_cognate", "non_cognate"}`. The 9
near-cognates are the single-substitution neighbours of ATG (matching the curated
manifest's negative codon set).

**Files:**
- Create: `src/tisiago/enumerate_codons.py`
- Test: `tests/test_enumerate_codons.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enumerate_codons.py
from tisiago.enumerate_codons import NEAR_COGNATES, classify_codon


def test_classify_aug():
    assert classify_codon("ATG") == "AUG"


def test_classify_near_cognates_are_single_mismatch_from_atg():
    # exactly the 9 single-substitution neighbours of ATG
    expected = {"CTG", "GTG", "TTG", "AAG", "ACG", "AGG", "ATA", "ATC", "ATT"}
    assert NEAR_COGNATES == expected
    for c in expected:
        assert classify_codon(c) == "near_cognate"


def test_classify_non_cognate():
    for c in ["AAA", "CCC", "GGG", "TTT", "GAT", "TAG"]:
        assert classify_codon(c) == "non_cognate"


def test_classify_lowercase_and_n():
    assert classify_codon("atg") == "AUG"          # case-insensitive
    assert classify_codon("ANG") == "non_cognate"  # ambiguous base -> non_cognate
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_enumerate_codons.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/tisiago/enumerate_codons.py
"""GTF-driven dense codon enumeration -> a scan manifest for the global caller.

Walks each transcript's exons (GENCODE v49 GTF) into spliced mRNA coordinates,
emits one row per reading position (every ``mrna_index``), reads its genome 3-mer,
classifies the codon, and labels it against the curated manifest's called TIS. The
output parquet is schema-compatible with ``extract.py`` (columns ``row_idx,
transcript_id, chrom, gstart, strand, codon``), so the existing GPU extraction +
``store.py`` assemble a dense scan store unchanged.

Coordinate conventions: GTF is 1-based inclusive; manifest ``gstart`` is 0-based.
The Task-4 correctness test pins the mapping by requiring enumerated ``(gstart,
codon)`` to equal the curated manifest for every existing candidate.
"""

from __future__ import annotations

# The 9 single-substitution neighbours of ATG (matches the curated negatives).
NEAR_COGNATES = {"CTG", "GTG", "TTG", "AAG", "ACG", "AGG", "ATA", "ATC", "ATT"}


def classify_codon(codon: str) -> str:
    """Classify a 3-mer as ``AUG`` / ``near_cognate`` / ``non_cognate``.

    Args:
        codon: a 3-letter codon (case-insensitive); non-ACGT bases -> non_cognate.

    Returns:
        ``"AUG"``, ``"near_cognate"``, or ``"non_cognate"``.
    """
    c = codon.upper()
    if c == "ATG":
        return "AUG"
    if c in NEAR_COGNATES:
        return "near_cognate"
    return "non_cognate"
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_enumerate_codons.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py && ruff format src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py
git add src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py
git commit -m "feat(enumerate): codon classification (AUG/near/non-cognate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Exon splicing → spliced-coordinate map (pure)

`spliced_positions` maps each mRNA index to the plus-strand genomic coordinate of that
base, for both strands. This is the correctness-critical geometry.

**Files:**
- Modify: `src/tisiago/enumerate_codons.py`
- Test: `tests/test_enumerate_codons.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_enumerate_codons.py
import numpy as np
from tisiago.enumerate_codons import spliced_positions


def test_spliced_positions_plus_strand_two_exons():
    # exons (0-based, half-open) [100,105) and [200,203): mRNA = 5 + 3 = 8 bases.
    # 5'->3' on + strand is ascending genomic order.
    exons = [(100, 105), (200, 203)]
    pos = spliced_positions(exons, "+")
    assert list(pos) == [100, 101, 102, 103, 104, 200, 201, 202]


def test_spliced_positions_minus_strand_reverses_and_descends():
    # same exons, - strand: mRNA 5'->3' walks high genomic -> low.
    exons = [(100, 105), (200, 203)]
    pos = spliced_positions(exons, "-")
    assert list(pos) == [202, 201, 200, 104, 103, 102, 101, 100]


def test_spliced_positions_length_matches_total_exon_length():
    exons = [(0, 10), (50, 60), (100, 130)]
    assert len(spliced_positions(exons, "+")) == 10 + 10 + 30
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_enumerate_codons.py::test_spliced_positions_plus_strand_two_exons -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

```python
# add to src/tisiago/enumerate_codons.py (add `import numpy as np` at the top)
import numpy as np


def spliced_positions(exons, strand: str) -> np.ndarray:
    """Plus-strand genomic coordinate of each base in mRNA 5'->3' order.

    Args:
        exons: list of (start, end) 0-based half-open genomic intervals (any order).
        strand: ``"+"`` or ``"-"``.

    Returns:
        int array of length = total exon length; element ``i`` is the 0-based
        genomic position of mRNA base ``i`` (reading 5'->3').
    """
    intervals = sorted(exons)  # ascending genomic
    coords = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in intervals])
    return coords if strand == "+" else coords[::-1].copy()
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_enumerate_codons.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py && ruff format src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py
git add src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py
git commit -m "feat(enumerate): exon splicing -> spliced genomic coordinate map

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: GTF parsing → per-transcript exon models (I/O)

`parse_gtf_exons` reads a GTF file and returns, for each transcript, its chrom, strand,
and exon intervals. Manual parsing (no library available).

**Files:**
- Modify: `src/tisiago/enumerate_codons.py`
- Test: `tests/test_enumerate_codons.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_enumerate_codons.py
from tisiago.enumerate_codons import parse_gtf_exons


def test_parse_gtf_exons(tmp_path):
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        '#comment line\n'
        'chr1\tHAVANA\ttranscript\t101\t300\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1.1";\n'
        'chr1\tHAVANA\texon\t101\t105\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1.1";\n'
        'chr1\tHAVANA\texon\t201\t203\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1.1";\n'
        'chr2\tHAVANA\texon\t51\t60\t.\t-\t.\tgene_id "G2"; transcript_id "ENST2.2";\n'
    )
    models = parse_gtf_exons(str(gtf), keep={"ENST1.1", "ENST2.2"})
    # GTF 1-based inclusive -> 0-based half-open: 101..105 -> [100,105)
    assert models["ENST1.1"] == {"chrom": "chr1", "strand": "+", "exons": [(100, 105), (200, 203)]}
    assert models["ENST2.2"] == {"chrom": "chr2", "strand": "-", "exons": [(50, 60)]}


def test_parse_gtf_exons_filters_by_keep(tmp_path):
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        'chr1\tHAVANA\texon\t1\t5\t.\t+\t.\ttranscript_id "KEEP.1";\n'
        'chr1\tHAVANA\texon\t1\t5\t.\t+\t.\ttranscript_id "DROP.1";\n'
    )
    models = parse_gtf_exons(str(gtf), keep={"KEEP.1"})
    assert set(models) == {"KEEP.1"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_enumerate_codons.py::test_parse_gtf_exons -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

```python
# add to src/tisiago/enumerate_codons.py (add `import re` at the top)
import re

_TX_RE = re.compile(r'transcript_id "([^"]+)"')


def parse_gtf_exons(gtf_path: str, keep: set[str] | None = None) -> dict:
    """Parse exon features from a GTF into per-transcript models.

    Args:
        gtf_path: path to a GENCODE/Ensembl GTF.
        keep: if given, only transcripts whose id is in this set are returned.

    Returns:
        dict ``transcript_id -> {"chrom", "strand", "exons"}`` where ``exons`` is a
        genomic-ascending list of 0-based half-open ``(start, end)`` intervals.
    """
    models: dict[str, dict] = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            mt = _TX_RE.search(f[8])
            if mt is None:
                continue
            tx = mt.group(1)
            if keep is not None and tx not in keep:
                continue
            start = int(f[3]) - 1  # 1-based inclusive -> 0-based
            end = int(f[4])        # inclusive end -> half-open
            rec = models.get(tx)
            if rec is None:
                models[tx] = {"chrom": f[0], "strand": f[6], "exons": [(start, end)]}
            else:
                rec["exons"].append((start, end))
    for rec in models.values():
        rec["exons"].sort()
    return models
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_enumerate_codons.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py && ruff format src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py
git add src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py
git commit -m "feat(enumerate): GTF exon parsing (1-based -> 0-based)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Assemble the scan manifest + correctness test against real data

The `main()` ties it together: pick the held-out transcripts from the curated manifest,
parse their exons, fetch spliced sequences via pyfaidx, enumerate every codon, assign
`gstart`/`codon_class`/`label_tis`/`split`, write `data/scan_manifest.parquet`. Then a
test runs the enumerator on a handful of real transcripts and asserts the coordinate
convention matches the curated manifest exactly.

**This task discovers the exact `gstart` convention empirically** — before writing `main`,
the implementer should inspect 3–5 known `+` and `-` candidates (their `gstart`, `strand`,
`mrna_index`, `codon`) and confirm the formula that reproduces `gstart`. The `+` strand is
`spliced_positions(...)[mrna_index]`. For `-`, determine whether the manifest `gstart`
equals that coordinate or `coordinate + 1` (consistent with `tiling.a_plus_of` using
`gstart - 1` for `-`). The test below is the acceptance criterion.

**Files:**
- Modify: `src/tisiago/enumerate_codons.py`
- Test: `tests/test_enumerate_codons.py`

- [ ] **Step 1: Install pyfaidx into the env**

```bash
uv pip install pyfaidx
```
Expected: pyfaidx installs (it's also in the `[extract]` extra).

- [ ] **Step 2: Write the failing integration test**

```python
# add to tests/test_enumerate_codons.py
import pandas as pd
import pytest
from tisiago.enumerate_codons import enumerate_transcript

GENOME = "/lab/barcheese01/mdiberna/swissisoform-v2/data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa"
GTF = "/lab/barcheese01/mdiberna/swissisoform-v2/data/reference/gencode.v49.primary_assembly.annotation.gtf"


@pytest.mark.skipif(not pd.io.common.file_exists(GENOME), reason="genome not present")
def test_enumerated_coords_match_curated_manifest():
    from pyfaidx import Fasta
    from tisiago.enumerate_codons import parse_gtf_exons

    man = pd.read_parquet("data/store/manifest.parquet")
    # take a few transcripts on a held-out chrom with both strands present
    sample_tx = (
        man[man.split == "test"]
        .drop_duplicates("transcript_id")
        .groupby("strand")
        .head(5)
        .transcript_id.tolist()
    )
    models = parse_gtf_exons(GTF, keep=set(sample_tx))
    fa = Fasta(GENOME, sequence_always_upper=True, rebuild=False)

    n_checked = 0
    for tx in sample_tx:
        if tx not in models:
            continue
        rows = enumerate_transcript(tx, models[tx], fa)  # DataFrame: mrna_index, gstart, codon, ...
        by_idx = rows.set_index("mrna_index")
        cur = man[(man.transcript_id == tx)]
        for _, c in cur.iterrows():
            if c.mrna_index not in by_idx.index:
                continue
            e = by_idx.loc[c.mrna_index]
            e = e.iloc[0] if hasattr(e, "iloc") and e.ndim > 1 else e
            assert int(e.gstart) == int(c.gstart), f"{tx} idx {c.mrna_index}: gstart {e.gstart} != {c.gstart}"
            assert e.codon == c.codon, f"{tx} idx {c.mrna_index}: codon {e.codon} != {c.codon}"
            n_checked += 1
    assert n_checked >= 20, f"only checked {n_checked} candidates"
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/test_enumerate_codons.py::test_enumerated_coords_match_curated_manifest -v`
Expected: FAIL — `ImportError: cannot import name 'enumerate_transcript'`.

- [ ] **Step 4: Implement `enumerate_transcript` + `main`**

Implement `enumerate_transcript(tx, model, fa)` returning a DataFrame with columns
`transcript_id, chrom, strand, mrna_index, gstart, codon, codon_class`. Use
`spliced_positions` for the plus-strand coordinate array; fetch the spliced mRNA sequence
by concatenating per-exon genome slices (revcomp on `-`); for each `i` in `0..L-3`, set
`codon = mrna_seq[i:i+3]`, `codon_class = classify_codon(codon)`, and `gstart` from the
empirically-confirmed convention (`+`: `coords[i]`; `-`: confirm `coords[i]` vs `coords[i]+1`).
Skip codons containing `N`.

```python
# add to src/tisiago/enumerate_codons.py (imports: argparse, pathlib.Path, pandas as pd)
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def enumerate_transcript(tx: str, model: dict, fa) -> "pd.DataFrame":
    """Enumerate every codon of one transcript into a DataFrame.

    Args:
        tx: transcript id.
        model: ``{"chrom", "strand", "exons"}`` from ``parse_gtf_exons``.
        fa: an open ``pyfaidx.Fasta`` over the genome.

    Returns:
        DataFrame with ``transcript_id, chrom, strand, mrna_index, gstart, codon,
        codon_class`` — one row per reading position ``0..L-3`` (codons with N skipped).
    """
    import pandas as pd

    chrom, strand, exons = model["chrom"], model["strand"], model["exons"]
    coords = spliced_positions(exons, strand)          # plus-strand pos per mRNA base
    seq = "".join(str(fa[chrom][s:e]) for s, e in exons)  # ascending-genomic seq
    mrna = seq if strand == "+" else _revcomp(seq)         # 5'->3'
    # gstart convention (confirmed empirically against the curated manifest):
    #   + : gstart = coords[i]
    #   - : gstart = coords[i] + 1   (so tiling.a_plus_of(gstart,'-') = gstart-1 = coords[i])
    g_off = 0 if strand == "+" else 1
    recs = []
    L = len(mrna)
    for i in range(L - 2):
        codon = mrna[i : i + 3]
        if "N" in codon:
            continue
        recs.append((tx, chrom, strand, i, int(coords[i]) + g_off, codon, classify_codon(codon)))
    return pd.DataFrame(
        recs,
        columns=["transcript_id", "chrom", "strand", "mrna_index", "gstart", "codon", "codon_class"],
    )


def main() -> None:
    import argparse
    from pathlib import Path

    import pandas as pd
    from pyfaidx import Fasta

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/store/manifest.parquet")
    ap.add_argument("--gtf", default="/lab/barcheese01/mdiberna/swissisoform-v2/data/reference/gencode.v49.primary_assembly.annotation.gtf")
    ap.add_argument("--genome", default="/lab/barcheese01/mdiberna/swissisoform-v2/data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default="data/scan_manifest.parquet")
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    sub = man[man.split.isin(args.splits)]
    tx_split = sub.drop_duplicates("transcript_id").set_index("transcript_id").split.to_dict()
    keep = set(tx_split)
    print(f"enumerating {len(keep)} transcripts from splits {args.splits}", flush=True)

    models = parse_gtf_exons(args.gtf, keep=keep)
    fa = Fasta(args.genome, sequence_always_upper=True, rebuild=False)
    pos_keys = set(zip(man.loc[man.label_tis == 1, "transcript_id"], man.loc[man.label_tis == 1, "mrna_index"]))

    frames = []
    for n, tx in enumerate(keep):
        if tx not in models:
            continue
        df = enumerate_transcript(tx, models[tx], fa)
        df["split"] = tx_split[tx]
        df["label_tis"] = [int((tx, i) in pos_keys) for i in df.mrna_index]
        frames.append(df)
        if n % 200 == 0:
            print(f"  {n}/{len(keep)}", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "row_idx", np.arange(len(out), dtype=np.int64))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    cls = out.codon_class.value_counts().to_dict()
    print(f"wrote {args.out}: {len(out)} positions, {out.transcript_id.nunique()} transcripts")
    print(f"  positives={int(out.label_tis.sum())}  codon_class={cls}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the correctness test (iterate on the `-` convention until it passes)**

Run: `pytest tests/test_enumerate_codons.py::test_enumerated_coords_match_curated_manifest -v`
Expected: PASS, ≥20 candidates checked. If `-` strand gstart mismatches by 1, flip `g_off`; if `+` mismatches, the bug is in `spliced_positions`/exon handling — debug there. Do not weaken the test.

- [ ] **Step 6: Produce the real scan manifest (CPU)**

Run: `python -m tisiago.enumerate_codons --splits val test --out data/scan_manifest.parquet`
Expected: writes `data/scan_manifest.parquet`; prints position count (millions), transcript count (~2,272), positives, and a codon_class breakdown dominated by `non_cognate`.

- [ ] **Step 7: Sanity-check the manifest**

Run:
```bash
python -c "
import pandas as pd, numpy as np
s=pd.read_parquet('data/scan_manifest.parquet'); m=pd.read_parquet('data/store/manifest.parquet')
assert (s.row_idx.values==np.arange(len(s))).all(), 'row_idx not contiguous'
# every curated test/val positive should appear as a positive in the scan
mp=set(zip(m[m.label_tis==1].transcript_id, m[m.label_tis==1].mrna_index))
sp=set(zip(s[s.label_tis==1].transcript_id, s[s.label_tis==1].mrna_index))
print('curated test+val positives covered by scan:', len(mp & sp), '/', len({(t,i) for t,i in mp if t in set(s.transcript_id)}))
print('scan positions:', len(s), 'positives:', int(s.label_tis.sum()))
"
```
Expected: row_idx contiguous; scan covers (essentially) all curated positives in the enumerated transcripts.

- [ ] **Step 8: Lint + commit (code only — `data/` is gitignored)**

```bash
ruff check src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py && ruff format src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py
git add src/tisiago/enumerate_codons.py tests/test_enumerate_codons.py
git commit -m "feat(enumerate): scan manifest assembly + coordinate correctness test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: SLURM wrapper for the dense scan extraction

`run_tis_scan.sh` mirrors `run_tis_extract.sh` but points `extract.py` at the scan
manifest and a separate parts dir. Written here, launched separately (GPU).

**Files:**
- Create: `scripts/run_tis_scan.sh`

- [ ] **Step 1: Read the existing wrapper to mirror it**

Run: `cat scripts/run_tis_extract.sh`
Note its `#SBATCH` lines, the `GEN=` genome path, the `--manifest/--tile-spec/--config/--genome/--out-dir/--shard-id/--n-shards` invocation, and the env-activation pattern.

- [ ] **Step 2: Create `scripts/run_tis_scan.sh`**

Copy `run_tis_extract.sh` verbatim, then change only:
- the `#SBATCH --job-name` to `${USER}_tis_scan`;
- the manifest argument to `data/scan_manifest.parquet`;
- the `--out-dir` to `data/scan_parts`;
- a comment header noting this embeds the dense scan manifest (test+val transcripts) and that the headline set is `ag16k` + `evo2_8k` (AlphaGenome 131k optional — restrict keys to bound the scan-store size).

Keep the same partitions, gres, shard logic, and `HF_HOME` export. Do not change resource requests.

- [ ] **Step 3: Shellcheck-lint and verify it is syntactically valid (do not submit)**

Run: `bash -n scripts/run_tis_scan.sh`
Expected: no syntax errors. (Do NOT `sbatch` it — the GPU run is launched separately after review.)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_tis_scan.sh
git commit -m "feat(scan): SLURM wrapper to embed the dense scan manifest

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `scan_eval.py` — global caller at true imbalance + grounding

Train+calibrate on the curated store (Phase 1 path), apply the head to the dense scan
store, report true-imbalance recall @ FP-budget on AUG+near-cognate codons and the
non-cognate grounding. The metric-combining logic is unit-tested on synthetic arrays;
real numbers await the GPU scan store.

**Files:**
- Create: `src/tisiago/scan_eval.py`
- Test: `tests/test_scan_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scan_eval.py
import numpy as np
from tisiago.scan_eval import grounding_stats


def test_grounding_stats_low_for_confident_negatives():
    p = np.full(1000, 0.01)
    res = grounding_stats(p, threshold=0.38)
    assert res["mean_p"] < 0.05
    assert res["fpr_at_threshold"] == 0.0
    assert res["n"] == 1000


def test_grounding_stats_flags_high_scores():
    p = np.concatenate([np.full(900, 0.01), np.full(100, 0.9)])
    res = grounding_stats(p, threshold=0.38)
    assert res["fpr_at_threshold"] == 0.1   # 100/1000 above threshold
    assert res["p95"] >= 0.9
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_scan_eval.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/tisiago/scan_eval.py
"""Global all-codon caller: apply the calibrated head to the dense scan store.

Trains + isotonic-calibrates the head on the curated store (Phase 1's
``caller.fit_calibrated_head``), then scores every enumerated codon in the dense
scan store. Reports recall at a false-positives-per-transcript budget over
AUG+near-cognate codons *at true genome-wide imbalance* (all enumerated negatives,
not the curated 3:1), and the non-cognate grounding (mean P -> expect ~0).

Run after the scan store is assembled (GPU extraction + store.py). CPU only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.caller import DEFAULT_KEYS, TRAIN_SUBSAMPLE, fit_calibrated_head, recall_at_fp_budget


def grounding_stats(p, threshold: float) -> dict:
    """Summarize non-cognate scores: they should be ~0 for a grounded predictor.

    Args:
        p: probabilities for held-out non-cognate codons.
        threshold: the caller's operating threshold (for FPR).

    Returns:
        dict with ``mean_p``, ``p95``, ``fpr_at_threshold``, ``n``.
    """
    p = np.asarray(p, dtype=np.float64)
    return {
        "mean_p": float(p.mean()),
        "p95": float(np.percentile(p, 95)),
        "fpr_at_threshold": float((p >= threshold).mean()),
        "n": int(p.size),
    }


def _load(keys, emb):
    return np.concatenate([np.load(emb / k).astype(np.float32) for k in keys], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curated-store", default="data/store")
    ap.add_argument("--scan-store", default="data/scan_store")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    ap.add_argument("--budget", type=float, default=1.0)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    cur = Path(args.curated_store)
    cm = pd.read_parquet(cur / "manifest.parquet")
    cy = cm.label_tis.values
    tr_all = np.where(cm.split.values == "train")[0]
    cal = np.where(cm.split.values == "val")[0]
    tr = rng.choice(tr_all, min(TRAIN_SUBSAMPLE, len(tr_all)), replace=False)
    Xc = _load(args.keys, cur / "embeddings")
    head = fit_calibrated_head(Xc[tr], cy[tr], Xc[cal], cy[cal])

    scan = Path(args.scan_store)
    sm = pd.read_parquet(scan / "manifest.parquet")
    Xs = _load(args.keys, scan / "embeddings")
    p = head["predict"](Xs)
    te = sm.split.values == "test"

    cognate = te & np.isin(sm.codon_class.values, ["AUG", "near_cognate"])
    noncog = te & (sm.codon_class.values == "non_cognate")
    res = recall_at_fp_budget(p[cognate], sm.label_tis.values[cognate], sm.transcript_id.values[cognate], budget=args.budget)
    ratio = (sm.label_tis.values[cognate] == 0).sum() / max(1, (sm.label_tis.values[cognate] == 1).sum())

    print(f"scan store: {len(sm)} positions  test cognate codons={int(cognate.sum())}  non-cognate={int(noncog.sum())}")
    print(f"true imbalance (neg:pos over cognate test codons) = {ratio:.1f}:1  (curated was 3:1)")
    print(f"\nCALLER @ ≤{args.budget} FP/transcript (true imbalance):")
    print(f"  recall={res['recall']:.3f}  threshold p≥{res['threshold']:.3f}  ({res['fp_per_transcript']:.3f} FP/transcript)")
    g = grounding_stats(p[noncog], threshold=res["threshold"])
    print(f"\nNON-COGNATE GROUNDING (held-out, never trained):")
    print(f"  mean p={g['mean_p']:.4f}  p95={g['p95']:.4f}  FPR@threshold={g['fpr_at_threshold']:.4f}  (expect ~0)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_scan_eval.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Confirm the full suite still passes**

Run: `pytest tests/ -q`
Expected: all tests pass (Phase 1 + Phase 2 unit tests).

- [ ] **Step 6: Lint + commit**

```bash
ruff check src/tisiago/scan_eval.py tests/test_scan_eval.py && ruff format src/tisiago/scan_eval.py tests/test_scan_eval.py
git add src/tisiago/scan_eval.py tests/test_scan_eval.py
git commit -m "feat(scan): global caller eval — true imbalance + non-cognate grounding

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Update ARCHITECTURE.md + CLAUDE.md

**Files:**
- Modify: `ARCHITECTURE.md`, `CLAUDE.md`

- [ ] **Step 1: Add the scan modules to the ARCHITECTURE.md §2 module map**

In `ARCHITECTURE.md` module-map code block, after the `caller.py` line add:
```
├── enumerate_codons.py  scan setup (CPU)   GTF + genome ──▶ dense scan manifest (every codon)
└── scan_eval.py         global caller (CPU) scan store ──▶ recall @ true imbalance · non-cognate≈0
```
And append rows to the module table:
```
| `enumerate_codons.py` | GTF→dense scan manifest of every codon (Phase 2) | numpy/pandas/pyfaidx | pure fns unit-tested; coords checked vs manifest |
| `scan_eval.py` | Apply calibrated head to the dense scan store (Phase 2) | numpy/pandas, `caller` | logic unit-tested; numbers need scan store |
```

- [ ] **Step 2: Add the scan_store layout note to CLAUDE.md**

In `CLAUDE.md` under "## Store layout", after the existing store tree add:
```
The Phase-2 dense scan produces a parallel `data/scan_store/` (same layout) row-aligned to
`data/scan_manifest.parquet` — every codon in held-out transcripts, scored by `scan_eval`.
```

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md CLAUDE.md
git commit -m "docs: add Phase 2 scan modules + scan_store layout

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** enumerator (`enumerate_codons.py`) → Tasks 1–4; correctness test vs manifest → Task 4 Step 5; reuse of `extract.py`/`store.py` via SLURM wrapper → Task 5; global-caller eval at true imbalance + non-cognate grounding → Task 6; doc updates → Task 7. Resolved decisions (scope=test+val, persist scan store, every-position) are realized in Task 4 `main` (`--splits val test`) and Task 5/6.
- **Type consistency:** `parse_gtf_exons` returns `{chrom,strand,exons}`; consumed identically in `enumerate_transcript` and Task-4 test. `enumerate_transcript` returns columns `transcript_id,chrom,strand,mrna_index,gstart,codon,codon_class`; `main` adds `split,label_tis,row_idx`; `scan_eval` reads `codon_class,label_tis,split,transcript_id`. `recall_at_fp_budget`/`fit_calibrated_head`/`DEFAULT_KEYS`/`TRAIN_SUBSAMPLE` imported from `caller.py` (defined Phase 1).
- **Placeholder scan:** none — all code complete. The one empirical step (the `-`-strand `g_off`) is bounded to a single value flip with the correctness test as oracle.
- **Out of scope (correctly):** the actual GPU extraction (launched separately via Task-5 wrapper), genome-wide scale-up, Phases 3–4.
- **Known risk surfaced:** scan-store size (tens of GB) — Task 5 notes restricting feature keys to the headline set to bound it.
