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

import re

import numpy as np

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
            end = int(f[4])  # inclusive end -> half-open
            rec = models.get(tx)
            if rec is None:
                models[tx] = {"chrom": f[0], "strand": f[6], "exons": [(start, end)]}
            else:
                rec["exons"].append((start, end))
    for rec in models.values():
        rec["exons"].sort()
    return models
