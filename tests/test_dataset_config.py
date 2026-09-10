import pytest
import yaml

from tisiago.dataset_config import DatasetConfig, load_dataset_config, write_provenance


def test_load_dataset_config_parses_required_fields(tmp_path):
    p = tmp_path / "dataset.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "dense_v1",
                "parts_dir": "data/scan_parts_allsplits",
                "out_store": "/ssd/tisiago_store/dense",
                "keys": ["evo2/W8k/blocks.28.mlp.l3/off0.npy"],
                "neg_cap": 1000000,
                "noncog_sample": 1000000,
                "seed": 0,
            }
        )
    )
    cfg = load_dataset_config(p)
    assert isinstance(cfg, DatasetConfig)
    assert cfg.name == "dense_v1" and cfg.neg_cap == 1_000_000
    assert cfg.keys == ["evo2/W8k/blocks.28.mlp.l3/off0.npy"]


def test_load_dataset_config_expands_environment_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TISIAGO_TEST_STORE", str(tmp_path / "store"))
    path = tmp_path / "dataset.yaml"
    path.write_text(
        "name: portable\n"
        "parts_dir: data/parts\n"
        "out_store: ${TISIAGO_TEST_STORE}\n"
        "keys: [feature.npy]\n"
        "neg_cap: 1\n"
        "noncog_sample: 1\n"
    )
    assert load_dataset_config(path).out_store == str(tmp_path / "store")


def test_load_dataset_config_rejects_unset_environment_path(tmp_path, monkeypatch):
    monkeypatch.delenv("TISIAGO_MISSING_STORE", raising=False)
    path = tmp_path / "dataset.yaml"
    path.write_text(
        "name: portable\n"
        "parts_dir: data/parts\n"
        "out_store: ${TISIAGO_MISSING_STORE}\n"
        "keys: [feature.npy]\n"
        "neg_cap: 1\n"
        "noncog_sample: 1\n"
    )
    with pytest.raises(ValueError, match="TISIAGO_MISSING_STORE"):
        load_dataset_config(path)


def test_write_provenance_merges_prior_keys(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    cfg = DatasetConfig(
        name="d",
        parts_dir="p",
        out_store=str(store),
        keys=["a.npy"],
        neg_cap=1,
        noncog_sample=1,
        seed=0,
    )
    write_provenance(store, cfg, {"a::b::c::off0": {"dim": 4096, "covered": 10}}, "sha1")
    write_provenance(store, cfg, {"x::y::z::off0": {"dim": 1536, "covered": 10}}, "sha1")
    cfg_out = yaml.safe_load((store / "config.yaml").read_text())
    assert set(cfg_out["keys"]) == {"a::b::c::off0", "x::y::z::off0"}  # accumulated
    assert cfg_out["git_sha"] == "sha1"
