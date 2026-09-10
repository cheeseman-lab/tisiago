"""Tests for cross-manifest representation identity checks."""

import numpy as np
import pandas as pd
import pytest

from tisiago.store_overlap import align_shared_sites, compare_overlap_feature


def _manifest(sites):
    return pd.DataFrame(
        [
            {
                "transcript_id": transcript,
                "mrna_index": position,
                "codon": "ATG",
                "label_tis": label,
            }
            for transcript, position, label in sites
        ]
    )


def test_align_shared_sites_uses_transcript_coordinates_and_checks_identity():
    reference = _manifest([("a", 1, 1), ("b", 2, 0), ("c", 3, 0)])
    candidate = _manifest([("c", 3, 0), ("a", 1, 1), ("d", 4, 0)])

    reference_rows, candidate_rows = align_shared_sites(reference, candidate)

    np.testing.assert_array_equal(reference_rows, [0, 2])
    np.testing.assert_array_equal(candidate_rows, [1, 0])


def test_compare_overlap_feature_is_chunked_and_exact(tmp_path):
    reference = np.arange(20, dtype=np.float16).reshape(5, 4)
    candidate = reference[[4, 2, 0, 3, 1]]
    np.save(tmp_path / "reference.npy", reference)
    np.save(tmp_path / "candidate.npy", candidate)

    metrics = compare_overlap_feature(
        tmp_path / "reference.npy",
        tmp_path / "candidate.npy",
        np.array([0, 1, 4]),
        np.array([2, 4, 0]),
        chunk_size=2,
    )

    assert metrics == {
        "n_rows": 3,
        "exact": True,
        "exact_fraction": 1.0,
        "max_abs": 0.0,
    }


def test_align_shared_sites_rejects_duplicate_sites():
    reference = _manifest([("a", 1, 1), ("a", 1, 0)])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        align_shared_sites(reference, _manifest([("a", 1, 1)]))
