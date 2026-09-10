"""Shared strand-aware DNA and transcript sequence helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyfaidx import Fasta

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return sequence.translate(_COMPLEMENT)[::-1]


def fetch_genomic_window(
    fasta: "Fasta", chrom: str, start: int, length: int, strand: str
) -> str:
    """Fetch a fixed genomic window, padding overruns and orienting it 5′→3′."""
    if strand not in {"+", "-"}:
        raise ValueError(f"invalid strand: {strand!r}")
    end = start + length
    chrom_len = len(fasta[chrom])
    clipped_start, clipped_end = max(0, start), min(chrom_len, end)
    core = str(fasta[chrom][clipped_start:clipped_end]).upper()
    sequence = (
        ("N" * (clipped_start - start))
        + core
        + ("N" * (end - clipped_end))
    )
    return reverse_complement(sequence) if strand == "-" else sequence


def spliced_transcript_sequence(fasta: "Fasta", model: dict) -> str:
    """Build an exon-spliced transcript sequence in biological 5′→3′ orientation."""
    strand = model["strand"]
    if strand not in {"+", "-"}:
        raise ValueError(f"invalid strand: {strand!r}")
    exons = sorted(model["exons"])
    genomic = "".join(
        str(fasta[model["chrom"]][start:end]) for start, end in exons
    ).upper()
    return reverse_complement(genomic) if strand == "-" else genomic
