import math

import numpy as np
import pandas as pd
import pytest

from tisiago.evaluation.start_metrics import M1_COLUMNS, m1_at_budget, m2_direction, top_k_calls
from tisiago.truth.differential import all_differential_starts


def _candidates() -> pd.DataFrame:
    rows = [
        ("K562", "T1", 10, "AUG", "annotated", 0.9),
        ("K562", "T1", 20, "other_near_cognate", "utr5", 0.8),
        ("K562", "T1", 30, "AGG", "cds_inframe", 0.5),
        ("K562", "T2", 100, "AUG", "annotated", 0.4),
        ("K562", "T2", 110, "AAG", "cds_outframe", 0.7),
        ("HeLa", "T1", 10, "AUG", "annotated", 0.2),
        ("HeLa", "T1", 20, "other_near_cognate", "utr5", 0.6),
        ("HeLa", "T1", 30, "AGG", "cds_inframe", 0.1),
    ]
    df = pd.DataFrame(
        rows, columns=["condition", "transcript_id", "gpos", "codon_class", "region", "score"]
    )
    return df.assign(chrom="chr7", strand="+")


def _truth() -> pd.DataFrame:
    rows = [
        ("K562", 10, "AUG", "annotated", True),
        ("K562", 20, "other_near_cognate", "utr5", False),  # not reproducible
        ("K562", 30, "AGG", "cds_inframe", True),
        ("K562", 100, "AUG", "annotated", True),
        ("K562", 500, "AUG", "annotated", True),  # never scored
        ("HeLa", 10, "AUG", "annotated", True),
        ("U2OS", 10, "AUG", "annotated", True),  # condition not evaluated
    ]
    df = pd.DataFrame(rows, columns=["condition", "gpos", "codon_class", "region", "reproducible"])
    return df.assign(chrom="chr7", strand="+")


def _row(table, k, condition, stratum) -> dict:
    sel = table[(table.k == k) & (table.condition == condition) & (table.stratum == stratum)]
    assert len(sel) == 1
    return sel.iloc[0].to_dict()


def _check(row, n_calls, n_tp, n_truth, n_recalled, n_unscored):
    assert (row["n_calls"], row["n_tp"], row["n_truth"]) == (n_calls, n_tp, n_truth)
    assert (row["n_recalled"], row["n_truth_unscored"]) == (n_recalled, n_unscored)
    prec = n_tp / n_calls if n_calls else math.nan
    rec = n_recalled / n_truth if n_truth else math.nan
    for got, want in ((row["precision"], prec), (row["recall"], rec)):
        assert (math.isnan(got) and math.isnan(want)) or got == pytest.approx(want)


def test_m1_k1_overall_and_by_condition():
    table = m1_at_budget(_candidates(), _truth(), k=1)
    assert list(table.columns) == list(M1_COLUMNS)
    # K562 calls: T1 -> 10 (TP), T2 -> 110 (FP). Truth covered: 10, 30, 100; 500 unscored.
    _check(_row(table, 1, "K562", "all"), 2, 1, 3, 1, 1)
    # HeLa calls: T1 -> 20 (FP). Truth: 10.
    _check(_row(table, 1, "HeLa", "all"), 1, 0, 1, 0, 0)
    _check(_row(table, 1, "all", "all"), 3, 1, 4, 1, 1)
    assert "U2OS" not in set(table.condition)


def test_m1_k1_strata():
    table = m1_at_budget(_candidates(), _truth(), k=1)
    _check(_row(table, 1, "K562", "codon_class:AUG"), 1, 1, 2, 1, 1)
    _check(_row(table, 1, "K562", "codon_class:AAG"), 1, 0, 0, 0, 0)  # recall NaN
    _check(_row(table, 1, "K562", "codon_class:AGG"), 0, 0, 1, 0, 0)  # precision NaN
    _check(_row(table, 1, "K562", "codon_class:non_cognate"), 0, 0, 0, 0, 0)
    _check(_row(table, 1, "K562", "cds:in"), 1, 0, 1, 0, 0)
    _check(_row(table, 1, "K562", "cds:out"), 1, 1, 2, 1, 1)
    assert math.isnan(_row(table, 1, "K562", "codon_class:AAG")["recall"])
    assert math.isnan(_row(table, 1, "K562", "codon_class:AGG")["precision"])


def test_m1_several_budgets():
    table = m1_at_budget(_candidates(), _truth(), k=[1, 2])
    assert sorted(set(table.k)) == [1, 2]
    # k=2 K562: 10, 20 (not reproducible -> FP), 110, 100.
    _check(_row(table, 2, "K562", "all"), 4, 2, 3, 2, 1)
    _check(_row(table, 2, "HeLa", "all"), 2, 1, 1, 1, 0)
    _check(_row(table, 2, "K562", "codon_class:other_near_cognate"), 1, 0, 0, 0, 0)


def test_top_k_ties_and_shared_sites():
    df = pd.DataFrame(
        {
            "condition": ["K562"] * 4,
            "transcript_id": ["T1", "T1", "T2", "T2"],
            "chrom": ["chr1"] * 4,
            "strand": ["+"] * 4,
            "gpos": [50, 40, 40, 60],
            "codon_class": ["AUG"] * 4,
            "region": ["annotated"] * 4,
            "score": [0.5, 0.5, 0.9, 0.1],
        }
    )
    calls = top_k_calls(df, 1)
    # T1 tie -> lower gpos 40; T2 -> 40 too, the same site: called once.
    assert calls.gpos.tolist() == [40]
    assert calls.score.tolist() == [0.9]
    with pytest.raises(ValueError):
        top_k_calls(df, 0)


def test_m1_rejects_bad_candidates():
    with pytest.raises(ValueError, match="missing"):
        m1_at_budget(_candidates().drop(columns=["region"]), _truth())
    with pytest.raises(ValueError, match="codon_class"):
        m1_at_budget(_candidates().assign(codon_class="near_cognate"), _truth())
    with pytest.raises(ValueError, match="NaN"):
        m1_at_budget(_candidates().assign(score=np.nan), _truth())


def _differential() -> pd.DataFrame:
    rows = [
        ("A", "B", 1, 1),
        ("A", "B", 4, -1),
        ("B", "C", 1, -1),
        ("B", "C", 4, 1),
        ("B", "C", 5, -1),
    ]
    df = pd.DataFrame(rows, columns=["condition_a", "condition_b", "gpos", "direction"])
    return df.assign(chrom="chr1", strand="+")


def _predictions() -> pd.DataFrame:
    rows = [("A", 1, 0.9), ("A", 4, 0.1), ("B", 1, 0.5), ("B", 4, 0.5), ("B", 5, 0.2)]
    rows += [("C", 1, 0.5), ("C", 4, 0.3)]  # C site 1 ties B (wrong); C site 5 missing
    df = pd.DataFrame(rows, columns=["condition", "gpos", "score"])
    return df.assign(chrom="chr1", strand="+")


def test_m2_direction_accuracy():
    table = m2_direction(_differential(), _predictions())
    assert list(table.columns)[:3] == ["condition_a", "condition_b", "n_starts"]
    rows = {(r.condition_a, r.condition_b): r for r in table.itertuples()}
    ab, bc, total = rows[("A", "B")], rows[("B", "C")], rows[("all", "all")]
    assert (ab.n_starts, ab.n_differential, ab.n_correct, ab.accuracy) == (2, 2, 2, 1.0)
    assert (bc.n_starts, bc.n_differential, bc.n_correct, bc.accuracy) == (2, 3, 1, 0.5)
    assert (total.n_starts, total.n_differential, total.n_correct) == (4, 5, 3)
    assert total.accuracy == 0.75
    assert table.iloc[0].condition_a == "all"


def test_m2_is_nan_not_zero_without_starts():
    empty = all_differential_starts(
        pd.DataFrame(
            {
                "condition": ["A", "A", "B", "B"],
                "rep": [1, 2, 1, 2],
                "chrom": ["chr1"] * 4,
                "strand": ["+"] * 4,
                "gpos": [1] * 4,
                "share": [0.5] * 4,
            }
        )
    )
    table = m2_direction(empty, _predictions())
    assert len(table) == 1
    assert table.iloc[0].n_starts == 0
    assert math.isnan(table.iloc[0].accuracy)
    unscored = m2_direction(_differential(), _predictions().iloc[:0])
    assert (unscored.n_starts == 0).all()
    assert unscored.accuracy.isna().all()


def test_m2_rejects_duplicate_predictions():
    with pytest.raises(ValueError, match="duplicate"):
        m2_direction(_differential(), pd.concat([_predictions(), _predictions()]))
