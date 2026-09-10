"""Tests for representation-comparison helpers."""

import numpy as np
import pandas as pd

from tisiago.representation_eval import (
    _metrics,
    load_feature_rows,
    near_neighbor_pairs,
    sample_training_rows,
)


def test_load_feature_rows_preserves_key_and_row_order(tmp_path):
    root = tmp_path / "embeddings"
    (root / "a").mkdir(parents=True)
    np.save(root / "a" / "x.npy", np.arange(12, dtype=np.float16).reshape(4, 3))
    np.save(root / "a" / "y.npy", np.arange(8, dtype=np.float16).reshape(4, 2))

    values, widths = load_feature_rows(root, ["a/x.npy", "a/y.npy"], np.array([1, 3]))

    assert widths == [3, 2]
    np.testing.assert_array_equal(values[:, :3], [[3, 4, 5], [9, 10, 11]])
    np.testing.assert_array_equal(values[:, 3:], [[2, 3], [6, 7]])


def test_near_neighbor_pairs_are_within_mature_transcript():
    manifest = pd.DataFrame(
        {
            "transcript_id": ["a", "a", "a", "b", "b"],
            "strand": ["+", "+", "+", "-", "-"],
            # Genomic distances cross introns and are intentionally much larger.
            "gstart": [100, 10_000, 300, 501, 50_000],
            "mrna_index": [10, 40, 210, 100, 150],
            "label_tis": [1, 0, 0, 1, 0],
        }
    )

    positive, negative, groups = near_neighbor_pairs(
        manifest, np.arange(len(manifest)), distance=64
    )

    assert list(zip(positive, negative, strict=True)) == [(0, 1), (3, 4)]
    assert [len(group[0]) for group in groups] == [1, 1]


def test_sample_training_rows_keeps_all_positives_and_caps_negatives():
    rows = np.arange(10)
    labels = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0])

    first = sample_training_rows(rows, labels, negative_cap=3, seed=4)
    repeated = sample_training_rows(rows, labels, negative_cap=3, seed=4)

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first[labels[first] == 1], [0, 3, 6])
    assert np.sum(labels[first] == 0) == 3


def test_sample_training_rows_rejects_single_class_input():
    try:
        sample_training_rows(np.arange(3), np.ones(3), negative_cap=2, seed=0)
        raise AssertionError("single-class training data should fail")
    except ValueError as error:
        assert "both classes" in str(error)


def test_metrics_reports_precision_at_fixed_threshold():
    metrics = _metrics(
        np.array([0.9, 0.8, 0.7, 0.1]),
        np.array([1, 0, 1, 0]),
        np.array(["a", "a", "b", "b"]),
        0.75,
        np.array([0]),
        np.array([1]),
    )

    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["fp_per_transcript"] == 0.5
