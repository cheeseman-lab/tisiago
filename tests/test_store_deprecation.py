import warnings
import tisiago.store as store


def test_assemble_emits_deprecation_for_dense(monkeypatch, tmp_path):
    # the dense-scan monolith path should warn; pass a sentinel that triggers the guard
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store.warn_if_dense_monolith(n_rows=62_683_645)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert any("scan_score" in str(x.message) for x in w)
