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


def test_stranded_frame_fraction_uses_same_strand_track():
    # - strand CDS [10, 16): codon starts are the highest coordinates 15 and 12.
    cds = [("chr1", "+", [(10, 16)]), ("chr1", "-", [(10, 16)])]
    plus = {"chr1": Counter({10: 3, 11: 1, 15: 50})}  # 15 is out of frame for + (index 5)
    minus = {"chr1": Counter({15: 2, 14: 2, 10: 40})}  # 10 is index 5 for - (out of frame)
    # + : 3 in frame of 3+1+50 = 54 ; - : 2 in frame of 2+2+40 = 44 ; combined count-weighted
    assert stranded_frame_fraction(plus, minus, cds) == pytest.approx(5 / 98)
