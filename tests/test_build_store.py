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
        parts / "s0.npz",
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
