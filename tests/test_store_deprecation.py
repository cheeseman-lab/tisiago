import warnings

import numpy as np
import yaml

import tisiago.store as store


def test_assemble_emits_deprecation_for_dense(monkeypatch, tmp_path):
    # the dense-scan monolith path should warn; pass a sentinel that triggers the guard
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store.warn_if_dense_monolith(n_rows=62_683_645)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert any("scan_score" in str(x.message) for x in w)


def test_atomic_store_writers_leave_complete_files(tmp_path):
    array_path = tmp_path / "features.npy"
    yaml_path = tmp_path / "config.yaml"

    store.atomic_save_array(array_path, np.arange(5))
    store.atomic_write_yaml(yaml_path, {"covered": 5})

    np.testing.assert_array_equal(np.load(array_path), np.arange(5))
    assert yaml.safe_load(yaml_path.read_text()) == {"covered": 5}
    assert not list(tmp_path.glob(".*.tmp"))
