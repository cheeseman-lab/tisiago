"""Tests for replicate concordance and CHX frame periodicity."""

from collections import Counter

import pytest

from tisiago.seq2func.qc import frame_fraction, replicate_concordance, stranded_frame_fraction


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


def test_minus_strand_frame_undoes_ribotish_shift():
    # - strand CDS [10, 16): codon first bases are 15 and 12; Ribo-TISH puts P-sites at +1.
    cds = [("chr1", "-", [(10, 16)])]
    counts = {"chr1": Counter({16: 5, 13: 3, 15: 2})}
    assert frame_fraction(counts, cds) == pytest.approx(8 / 10)


def test_stranded_frame_fraction_uses_same_strand_track():
    cds = [("chr1", "+", [(10, 16)]), ("chr1", "-", [(10, 16)])]
    plus = {"chr1": Counter({10: 3, 11: 1, 15: 50})}  # 15 is out of frame for + (index 5)
    minus = {"chr1": Counter({16: 2, 15: 2, 11: 40})}  # 16 in frame (shifted); 11 is index 4 for -
    # + : 3 in frame of 54 ; - : 2 in frame of 44 ; plus-track counts must not leak into minus
    assert stranded_frame_fraction(plus, minus, cds) == pytest.approx(5 / 98)
