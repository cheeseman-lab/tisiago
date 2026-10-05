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
