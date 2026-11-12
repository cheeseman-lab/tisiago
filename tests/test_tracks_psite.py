"""Tests for Ribo-TISH-identical P-site placement."""

import pysam

from tisiago.tracks.psite import psite, ribotish_genome_pos, rna_end5

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
