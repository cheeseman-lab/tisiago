import numpy as np
import pandas as pd
import pytest

from tisiago.truth.differential import (
    all_differential_starts,
    count_differential,
    differential_starts,
    usage_shares,
)

# share[condition][rep] for five sites (gpos 1..5) on chr1 +.
SHARES = {
    "A": {1: [0.6, 0.5, 0.5, 0.2, 0.3], 2: [0.7, 0.2, 0.55, 0.3, np.nan]},
    "B": {1: [0.3, 0.3, 0.5, 0.5, 0.1], 2: [0.4, 0.4, 0.5, 0.5, 0.1]},
    "C": {1: [0.6, 0.5, 0.5, 0.2, 0.3], 2: [0.7, 0.2, 0.55, 0.3, 0.3]},
}


def _shares() -> pd.DataFrame:
    rows = []
    for condition, reps in SHARES.items():
        for rep, values in reps.items():
            for gpos, share in enumerate(values, start=1):
                rows.append((condition, rep, "chr1", "+", gpos, share))
    return pd.DataFrame(rows, columns=["condition", "rep", "chrom", "strand", "gpos", "share"])


def test_differential_starts_requires_both_pairings():
    # A - B: site 1 (+0.3, +0.3) up; site 2 (+0.2, -0.2) flips; site 3 (0, +0.05) small;
    # site 4 (-0.3, -0.2) down; site 5 untested (A rep2 NaN).
    hit = differential_starts(_shares(), "A", "B", min_delta=0.1)
    assert hit.gpos.tolist() == [1, 4]
    assert hit.direction.tolist() == [1, -1]
    assert hit.delta.round(6).tolist() == [0.3, -0.25]
    assert set(hit.columns) >= {"condition_a", "condition_b", "delta_1_1", "delta_2_2"}


def test_threshold_applies_to_every_pairing():
    hit = differential_starts(_shares(), "A", "B", min_delta=0.25)
    assert hit.gpos.tolist() == [1]


def test_crossed_pairings_can_be_required():
    hit = differential_starts(
        _shares(), "A", "B", min_delta=0.1, pairings=((1, 1), (2, 2), (1, 2), (2, 1))
    )
    # Site 4 crossed: A1 - B2 = -0.3, A2 - B1 = -0.2 -> still down; site 1: 0.2, 0.4 -> up.
    assert hit.gpos.tolist() == [1, 4]


def test_count_differential_per_pair():
    counts = count_differential(_shares(), min_delta=0.1)
    assert counts[["condition_a", "condition_b"]].values.tolist() == [
        ["A", "B"],
        ["A", "C"],
        ["B", "C"],
    ]
    ab, ac, bc = counts.to_dict("records")
    fields = ["n_tested", "n_a_higher", "n_b_higher", "n_differential"]
    assert [ab[f] for f in fields] == [4, 1, 1, 2]
    assert (ac["n_tested"], ac["n_differential"]) == (4, 0)
    # B - C: site 1 (-0.3, -0.3) C higher; site 4 (+0.3, +0.2) B higher; site 5 (-0.2, -0.2).
    assert [bc[f] for f in fields] == [5, 1, 2, 3]


def test_all_differential_starts_concatenates_pairs():
    hits = all_differential_starts(_shares(), conditions=["A", "B", "C"], min_delta=0.1)
    assert len(hits) == 5
    assert hits.groupby(["condition_a", "condition_b"]).size().to_dict() == {
        ("A", "B"): 2,
        ("B", "C"): 3,
    }
    empty = all_differential_starts(_shares(), conditions=["A", "C"])
    assert empty.empty and "direction" in empty.columns


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="missing"):
        differential_starts(_shares().drop(columns=["share"]), "A", "B")
    with pytest.raises(ValueError, match="duplicate"):
        differential_starts(pd.concat([_shares(), _shares()]), "A", "B")
    with pytest.raises(ValueError, match="no shares"):
        differential_starts(_shares(), "A", "B", pairings=((1, 3),))


def test_usage_shares_within_gene():
    counts = pd.DataFrame(
        {
            "condition": ["A"] * 4,
            "rep": [1, 1, 1, 2],
            "gene": ["g1", "g1", "g2", "g2"],
            "chrom": ["chr1"] * 4,
            "strand": ["+"] * 4,
            "gpos": [1, 2, 3, 3],
            "count": [3, 1, 0, 5],
        }
    )
    shares = usage_shares(counts)
    assert shares.share.tolist()[:2] == [0.75, 0.25]
    assert np.isnan(shares.share.iloc[2])  # gene with no reads
    assert shares.share.iloc[3] == 1.0
