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
