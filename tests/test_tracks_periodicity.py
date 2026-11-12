"""Tests for the CHX codon-phase helpers (synthetic arrays, no model)."""

import numpy as np

from tisiago.tracks.periodicity import dominant_phases, phase_counts, region_positions


def _periodic(length=300, phase=0, weights=(6.0, 1.0, 1.0)):
    out = np.zeros(length)
    for k, w in enumerate(weights):
        out[(phase + k) % 3 :: 3] = w
    return out


def test_region_positions_plus_and_minus():
    pos, ranks = region_positions("+", [(20, 23), (10, 12)])
    assert pos.tolist() == [10, 11, 20, 21, 22]
    assert ranks.tolist() == [0, 0, 1, 1, 1]
    pos, ranks = region_positions("-", [(10, 12), (20, 23)])
    assert pos.tolist() == [22, 21, 20, 11, 10]
    assert ranks.tolist() == [0, 0, 0, 1, 1]


def test_phase_counts_trims_and_pools():
    counts = phase_counts(np.arange(14), block=2)
    assert counts.tolist() == [[0 + 3, 1 + 4, 2 + 5], [6 + 9, 7 + 10, 8 + 11]]


def test_dominant_phases_marks_ties_and_empty_blocks():
    values = np.concatenate([_periodic(30, phase=2), np.zeros(30), np.ones(30)])
    assert dominant_phases(values, 10).tolist() == [2, -1, -1]


def test_dominant_phases_count_floor_applies_to_block_total():
    values = np.concatenate([_periodic(30, weights=(2.0, 1.0, 0.0)), _periodic(30)])
    # block totals 30 and 80
    assert dominant_phases(values, 10, min_counts=30).tolist() == [0, 0]
    assert dominant_phases(values, 10, min_counts=31).tolist() == [-1, 0]
