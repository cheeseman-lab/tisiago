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
