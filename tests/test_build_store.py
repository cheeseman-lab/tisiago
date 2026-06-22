import numpy as np
import pandas as pd
import yaml

from scripts.build_store import build_store
from tisiago.dataset_config import DatasetConfig


def test_build_store_gathers_to_compact_with_provenance(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    out = tmp_path / "store"
    out.mkdir()
    key = "ag::L16k::decoder_1bp::off0"
    # manifest: 3 wanted compact rows from src {2,5,9}
    pd.DataFrame({"src_row_idx": [2, 5, 9], "row_idx": [0, 1, 2]}).to_parquet(
        out / "manifest.parquet"
    )
    np.savez(
        parts / "a_shard0.npz",
        row_idx=np.array([2, 5, 9]),
        **{key: np.arange(6, dtype=np.float16).reshape(3, 2)},
    )
    cfg = DatasetConfig(
        name="t",
        parts_dir=str(parts),
        out_store=str(out),
        keys=["ag/L16k/decoder_1bp/off0.npy"],
        neg_cap=1,
        noncog_sample=1,
        seed=0,
    )
    build_store(cfg, "deadbeef")
    arr = np.load(out / "embeddings/ag/L16k/decoder_1bp/off0.npy")
    assert arr.shape == (3, 2) and arr[2, 0] == 4  # src9 -> compact row 2
    prov = yaml.safe_load((out / "config.yaml").read_text())
    assert prov["git_sha"] == "deadbeef" and prov["keys"][key]["covered"] == 3


def test_build_store_two_segment_key(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    out = tmp_path / "store"
    out.mkdir()
    key = "onehot::kozakW20"
    pd.DataFrame({"src_row_idx": [0, 1], "row_idx": [0, 1]}).to_parquet(
        out / "manifest.parquet"
    )
    np.savez(
        parts / "a_shard0.npz",
        row_idx=np.array([0, 1]),
        **{key: np.arange(4, dtype=np.float16).reshape(2, 2)},
    )
    cfg = DatasetConfig(
        name="t",
        parts_dir=str(parts),
        out_store=str(out),
        keys=["onehot/kozakW20.npy"],
        neg_cap=1,
        noncog_sample=1,
        seed=0,
    )
    build_store(cfg, "abc")
    arr = np.load(out / "embeddings/onehot/kozakW20.npy")
    assert arr.shape == (2, 2)


def test_build_store_ignores_non_shard_npz(tmp_path):
    # a predictions .npz (no row_idx) sharing the parts dir must NOT be read
    parts = tmp_path / "parts"
    parts.mkdir()
    out = tmp_path / "store"
    out.mkdir()
    key = "ag::L16k::decoder_1bp::off0"
    pd.DataFrame({"src_row_idx": [2, 5, 9], "row_idx": [0, 1, 2]}).to_parquet(
        out / "manifest.parquet"
    )
    np.savez(
        parts / "a_shard0.npz",
        row_idx=np.array([2, 5, 9]),
        **{key: np.arange(6, dtype=np.float16).reshape(3, 2)},
    )
    # output file with no row_idx member, in the same dir — would KeyError if read as a shard
    np.savez(parts / "dense_ag_preds.npz", p=np.zeros(3), y=np.zeros(3))
    cfg = DatasetConfig(
        name="t",
        parts_dir=str(parts),
        out_store=str(out),
        keys=["ag/L16k/decoder_1bp/off0.npy"],
        neg_cap=1,
        noncog_sample=1,
        seed=0,
    )
    build_store(cfg, "sha")  # must not raise
    assert (out / "embeddings/ag/L16k/decoder_1bp/off0.npy").exists()
