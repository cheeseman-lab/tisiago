"""Truth-table schema and the pure tagging rules (codon class, region, split, line group).

Coordinates: ``gpos`` is the 0-based genomic base of the start codon's first transcribed base
(the A of ATG). On + the codon is ``gpos..gpos+2``; on - it is ``gpos-2..gpos``. Ribo-TISH's
``Locus`` and the manifest's ``gstart`` are ``gpos`` on + and ``gpos + 1`` on - (half-open end);
:func:`gpos_from_locus` converts. ``tx_pos`` is the 0-based position of that base on the
transcript (5' -> 3'), Ribo-TISH's ``Start``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

CODON_CLASSES = ("AUG", "AAG", "AGG", "other_near_cognate", "non_cognate")
REGIONS = ("utr5", "annotated", "cds_inframe", "cds_outframe", "other")
IN_CDS = ("cds_inframe", "cds_outframe")
NEAR_COGNATES = frozenset({"CTG", "GTG", "TTG", "AAG", "ACG", "AGG", "ATA", "ATC", "ATT"})
STRANDS = ("+", "-")
SPLITS = {"chr7": "val", "chr8": "test", "chr9": "test"}
LINE_GROUPS = {"RPE1": "RPE1"}
SITE_KEY = ("chrom", "strand", "gpos")
START_KEY = ("condition", *SITE_KEY)
TRUTH_COLUMNS = (
    "condition",
    "line",
    "transcript_id",
    "gene",
    "chrom",
    "strand",
    "gpos",
    "tx_pos",
    "codon",
    "codon_class",
    "region",
    "pause",
    "rep1",
    "rep2",
    "reproducible",
    "split",
)
_BASES = frozenset("ACGT")


def normalize_codon(codon: str) -> str:
    """Upper-case DNA 3-mer (``U`` -> ``T``); raises ``ValueError`` if it is not one."""
    if not isinstance(codon, str):
        raise ValueError(f"codon must be a string, got {codon!r}")
    c = codon.strip().upper().replace("U", "T")
    if len(c) != 3 or not set(c) <= _BASES:
        raise ValueError(f"invalid codon {codon!r}")
    return c


def codon_class(codon: str) -> str:
    """``AUG`` / ``AAG`` / ``AGG`` / ``other_near_cognate`` / ``non_cognate`` of a 3-mer.

    Near-cognates are the nine codons one substitution from AUG; AAG and AGG are kept apart
    because harringtonine arrests elongation on them (spec §8).
    """
    c = normalize_codon(codon)
    if c == "ATG":
        return "AUG"
    if c in ("AAG", "AGG"):
        return c
    if c in NEAR_COGNATES:
        return "other_near_cognate"
    return "non_cognate"


def region_of(tx_pos: int, cds_start: float | None, cds_end: float | None) -> str:
    """Region of a start at transcript position ``tx_pos`` given the transcript's CDS.

    Args:
        tx_pos: 0-based transcript position of the codon's first base.
        cds_start: transcript position of the annotated start codon's first base, or
            None/NaN for a transcript without a CDS.
        cds_end: transcript position one past the stop codon's last base.

    Returns:
        ``utr5`` (upstream of the CDS), ``annotated`` (the CDS start), ``cds_inframe`` /
        ``cds_outframe`` (inside the CDS, by frame relative to the CDS start), or ``other``
        (no CDS, or at/after the CDS end).
    """
    if cds_start is None or cds_end is None or pd.isna(cds_start) or pd.isna(cds_end):
        return "other"
    cs, ce = int(cds_start), int(cds_end)
    if tx_pos < cs:
        return "utr5"
    if tx_pos == cs:
        return "annotated"
    if tx_pos < ce:
        return "cds_inframe" if (tx_pos - cs) % 3 == 0 else "cds_outframe"
    return "other"


def parse_blocks(value) -> list[int]:
    """Exon starts or ends from a ``,``-separated string or a sequence of ints."""
    if isinstance(value, str):
        return [int(v) for v in value.strip().strip(",").split(",") if v != ""]
    return [int(v) for v in value]


def genome_to_tx(
    gpos: int, strand: str, exon_starts: Sequence[int], exon_ends: Sequence[int]
) -> int | None:
    """Transcript position of genomic base ``gpos``, or None if it is not exonic.

    Exons are 0-based half-open ``[start, end)`` in any order; the transcript runs 5' -> 3'
    (ascending on +, descending on -).
    """
    exons = sorted(zip(parse_blocks(exon_starts), parse_blocks(exon_ends), strict=True))
    if strand == "-":
        exons = exons[::-1]
    elif strand != "+":
        raise ValueError(f"invalid strand {strand!r}")
    offset = 0
    for start, end in exons:
        if start <= gpos < end:
            return offset + (gpos - start if strand == "+" else end - 1 - gpos)
        offset += end - start
    return None


def cds_tx_bounds(
    strand: str,
    exon_starts: Sequence[int],
    exon_ends: Sequence[int],
    cds_start: float | None,
    cds_end: float | None,
) -> tuple[int, int] | None:
    """Transcript-coordinate ``(start, end)`` of a genomic CDS, or None without a CDS.

    ``cds_start`` / ``cds_end`` are genomic, lowest coordinate and half-open end, start and
    stop codons included (the release ``transcripts`` table and the GTF convention).
    """
    if cds_start is None or cds_end is None or pd.isna(cds_start) or pd.isna(cds_end):
        return None
    lo, hi = int(cds_start), int(cds_end) - 1
    first, last = (lo, hi) if strand == "+" else (hi, lo)
    a = genome_to_tx(first, strand, exon_starts, exon_ends)
    b = genome_to_tx(last, strand, exon_starts, exon_ends)
    if a is None or b is None:
        raise ValueError(f"CDS bounds {cds_start}-{cds_end} are not exonic")
    return a, b + 1


def gpos_from_locus(locus, strand) -> np.ndarray:
    """``gpos`` from a Ribo-TISH ``Locus`` / manifest ``gstart`` (minus strand is the end)."""
    locus = np.asarray(locus, dtype=np.int64)
    return np.where(np.asarray(strand) == "-", locus - 1, locus)


def gstart_from_gpos(gpos, strand) -> np.ndarray:
    """Manifest ``gstart`` (and Ribo-TISH ``Locus``) of each start; inverse of the above."""
    gpos = np.asarray(gpos, dtype=np.int64)
    return np.where(np.asarray(strand) == "-", gpos + 1, gpos)


def split_of(chrom: str) -> str:
    """Frozen split of a chromosome: val chr7, test chr8/chr9, train otherwise."""
    return SPLITS.get(str(chrom), "train")


def line_of(condition: str) -> str:
    """Leave-one-line-out group of a condition (RPE1 states are one group)."""
    head = str(condition).split("_", 1)[0]
    return LINE_GROUPS.get(head, str(condition))


def tag_regions(starts: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    """Compute ``tx_pos`` and ``region`` of each start on its own transcript.

    Args:
        starts: rows with ``transcript_id``, ``chrom``, ``strand``, ``gpos``.
        annotation: one row per transcript with ``transcript_id``, ``chrom``, ``strand``,
            ``exon_starts``, ``exon_ends``, ``cds_start``, ``cds_end`` (genomic; see
            :func:`cds_tx_bounds`).

    Returns:
        DataFrame aligned to ``starts`` with ``tx_pos`` and ``region``.

    Raises:
        ValueError: a transcript is missing from the annotation, disagrees on chrom/strand, or
            does not contain its start.
    """
    ann = annotation.drop_duplicates("transcript_id").set_index("transcript_id")
    missing = set(starts.transcript_id.astype(str)) - set(ann.index.astype(str))
    if missing:
        example = sorted(missing)[:3]
        raise ValueError(f"{len(missing)} transcripts missing from annotation, e.g. {example}")
    cache: dict[str, tuple] = {}
    tx_pos = np.empty(len(starts), dtype=np.int64)
    regions = np.empty(len(starts), dtype=object)
    rows = zip(starts.transcript_id, starts.chrom, starts.strand, starts.gpos, strict=True)
    for i, (tid, chrom, strand, gpos) in enumerate(rows):
        if tid not in cache:
            a = ann.loc[tid]
            exons = (parse_blocks(a.exon_starts), parse_blocks(a.exon_ends))
            cds = cds_tx_bounds(a.strand, *exons, a.cds_start, a.cds_end)
            cache[tid] = (str(a.chrom), str(a.strand), exons, cds)
        a_chrom, a_strand, exons, cds = cache[tid]
        if (str(chrom), str(strand)) != (a_chrom, a_strand):
            raise ValueError(f"{tid}: start on {chrom}{strand}, transcript on {a_chrom}{a_strand}")
        pos = genome_to_tx(int(gpos), a_strand, *exons)
        if pos is None:
            raise ValueError(f"{tid}: start {chrom}:{gpos}:{strand} is not exonic")
        tx_pos[i] = pos
        regions[i] = region_of(pos, *(cds if cds is not None else (None, None)))
    return pd.DataFrame({"tx_pos": tx_pos, "region": regions}, index=starts.index)
