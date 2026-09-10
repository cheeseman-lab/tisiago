"""Tests for embedding-shard artifact comparison."""

from __future__ import annotations

import numpy as np
import pytest

from tisiago.compare_extractions import compare_extraction_artifacts


def _write_artifact(path, *, rows=(2, 7), offset=0.0):
    features = np.asarray([[1000.0, -1000.0], [2.0, 4.0]], dtype=np.float16)
    np.savez(
        path,
        row_idx=np.asarray(rows, dtype=np.int64),
        **{"evo2::TXP::layer::off0": features + np.float16(offset)},
    )


def test_compare_extraction_artifacts_reports_upcast_drift(tmp_path):
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    _write_artifact(reference)
    _write_artifact(candidate, offset=1.0)

    result = compare_extraction_artifacts(reference, candidate)

    metrics = result["evo2::TXP::layer::off0"]
    assert metrics["allclose"] is False
    assert metrics["max_abs"] == 1.0
    assert np.isfinite(metrics["rmse"])


def test_compare_extraction_artifacts_requires_row_alignment(tmp_path):
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    _write_artifact(reference)
    _write_artifact(candidate, rows=(7, 2))

    with pytest.raises(AssertionError, match="row_idx"):
        compare_extraction_artifacts(reference, candidate)


def test_compare_extraction_artifacts_handles_empty_shards(tmp_path):
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    for path in (reference, candidate):
        np.savez(
            path,
            row_idx=np.empty(0, dtype=np.int64),
            **{"evo2::TXP::layer::off0": np.empty((0, 2), dtype=np.float16)},
        )

    metrics = compare_extraction_artifacts(reference, candidate)

    assert metrics["evo2::TXP::layer::off0"]["allclose"] is True
    assert metrics["evo2::TXP::layer::off0"]["max_abs"] == 0.0
