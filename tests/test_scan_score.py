
import numpy as np
import pandas as pd

from tisiago.scan_score import score_shards


def test_score_shards_aligns_to_manifest_via_streaming(tmp_path):
    # 3 wanted compact rows mapped from src {2,5,9}; one 2-col key
    key = "ag::L16k::decoder_1bp::off0"
    manifest = pd.DataFrame({"src_row_idx": [2, 5, 9], "row_idx": [0, 1, 2]})
    # shard A holds src 2 and 9, shard B holds src 5 (plus unwanted src 7)
    np.savez(
        tmp_path / "a.npz",
        row_idx=np.array([2, 9]),
        **{key: np.array([[1.0, 0.0], [3.0, 0.0]], np.float16)},
    )
    np.savez(
        tmp_path / "b.npz",
        row_idx=np.array([5, 7]),
        **{key: np.array([[2.0, 0.0], [9.0, 0.0]], np.float16)},
    )
    # head: P = first feature column (so we can assert ordering)
    head = {"predict": lambda X: X[:, 0]}
    p = score_shards(tmp_path, "*.npz", manifest, [key + ".npy"], head)
    # compact row 0<-src2=1.0, row1<-src5=2.0, row2<-src9=3.0
    assert p.tolist() == [1.0, 2.0, 3.0]


def test_score_shards_raises_on_incomplete_coverage(tmp_path):
    key = "ag::L16k::decoder_1bp::off0"
    manifest = pd.DataFrame({"src_row_idx": [2, 5, 9], "row_idx": [0, 1, 2]})
    np.savez(
        tmp_path / "a.npz",
        row_idx=np.array([2, 9]),
        **{key: np.ones((2, 2), np.float16)},
    )  # src 5 never appears
    head = {"predict": lambda X: X[:, 0]}
    try:
        score_shards(tmp_path, "*.npz", manifest, [key + ".npy"], head)
        assert False, "expected coverage assertion"
    except AssertionError as e:
        assert "covered" in str(e)
