"""Tests for the Ribo-TISH command, Jimmy's filter, and call comparison."""

import pandas as pd

from tisiago.seq2func.ribotish import (
    call_keys,
    compare_calls,
    filter_predictions,
    predict_command,
)


def test_predict_command_matches_jimmys_flags():
    argv = predict_command(
        "ribotish", ["t1.bam", "t2.bam"], ["c1.bam", "c2.bam"], ["t1.p", "t2.p"],
        ["c1.p", "c2.p"], "g.gtf", "g.fa", "out.txt", "bg.txt",
    )
    assert argv == [
        "ribotish", "predict", "-t", "t1.bam,t2.bam", "-b", "c1.bam,c2.bam",
        "--tispara", "t1.p,t2.p", "--ribopara", "c1.p,c2.p", "-g", "g.gtf", "-f", "g.fa",
        "-o", "out.txt", "-e", "bg.txt", "--minaalen", "3", "--alt", "--seq", "--aaseq",
        "--verbose",
    ]


def _calls(rows):
    return pd.DataFrame(rows, columns=["Tid", "Start", "TISPvalue", "RiboPvalue", "FisherQvalue"])


def test_filter_predictions_uses_jimmys_thresholds():
    df = _calls([("a", 1, 0.01, 0.01, 0.05), ("b", 2, 0.02, 0.0, 0.0),
                 ("c", 3, 0.0, 0.02, 0.0), ("d", 4, 0.0, 0.0, 0.06)])
    assert list(filter_predictions(df).Tid) == ["a"]


def test_compare_calls_jaccard():
    observed = _calls([("a", 1, 0, 0, 0), ("b", 2, 0, 0, 0)])
    reference = _calls([("a", 1, 0, 0, 0), ("c", 3, 0, 0, 0)])
    assert call_keys(observed) == {("a", 1), ("b", 2)}
    assert compare_calls(observed, reference) == {
        "n_observed": 2, "n_reference": 2, "n_shared": 1, "jaccard": 1 / 3,
    }
