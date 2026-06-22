
import numpy as np
import pandas as pd

from tisiago.shard_io import build_src_to_compact, iter_shards, map_rows


def _write_shard(p, rows, key, mat):
    np.savez(p, row_idx=rows, **{key: mat})


def test_map_rows_masks_out_of_range_and_unwanted():
    # wanted src rows {2,5,9} -> compact {0,1,2}
    m = pd.DataFrame({"src_row_idx": [2, 5, 9], "row_idx": [0, 1, 2]})
    src2cmp = build_src_to_compact(m)
    rows = np.array([2, 3, 9, 99])          # 3 unwanted, 99 out-of-range
    keep, dst = map_rows(rows, src2cmp)
    assert keep.tolist() == [True, False, True, False]
    assert dst.tolist() == [0, 2]


def test_iter_shards_yields_members_without_row_idx(tmp_path):
    k = "evo2::W8k::blocks.28.mlp.l3::off0"
    _write_shard(tmp_path / "s0.npz", np.array([2, 5]), k, np.ones((2, 4), np.float16))
    out = list(iter_shards(tmp_path, "s*.npz"))
    assert len(out) == 1
    rows, members = out[0]
    assert rows.tolist() == [2, 5]
    assert list(members) == [k] and members[k].shape == (2, 4)
