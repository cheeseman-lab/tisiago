import numpy as np
import pytest

from tisiago.extraction_io import PreallocatedShard, shard_is_complete


def test_preallocated_shard_writes_rows_atomically(tmp_path):
    rows = np.array([2, 5, 9])
    shard = PreallocatedShard(rows)
    shard.add("model::layer", [9, 2], [[9.0, 1.0], [2.0, 1.0]])
    shard.add("model::layer", [5], [[5.0, 1.0]])
    out = tmp_path / "part.npz"
    shard.save(out, expected_keys={"model::layer"})

    assert shard_is_complete(out, rows, {"model::layer"})
    with np.load(out) as saved:
        assert saved["row_idx"].tolist() == [2, 5, 9]
        assert saved["model::layer"][:, 0].tolist() == [2.0, 5.0, 9.0]
        assert saved["model::layer"].dtype == np.float16
    assert not list(tmp_path.glob(".*.tmp"))


def test_preallocated_shard_rejects_missing_and_duplicate_rows(tmp_path):
    shard = PreallocatedShard([1, 2])
    shard.add("key", [1], [[1.0]])
    with pytest.raises(ValueError, match="covered 1/2"):
        shard.save(tmp_path / "incomplete.npz", expected_keys={"key"})
    with pytest.raises(ValueError, match="duplicate"):
        shard.add("key", [1], [[2.0]])


def test_preallocated_shard_rejects_nonfinite_features(tmp_path):
    shard = PreallocatedShard([1])
    shard.add("key", [1], [[np.nan]])
    with pytest.raises(ValueError, match="non-finite"):
        shard.save(tmp_path / "invalid.npz", expected_keys={"key"})
