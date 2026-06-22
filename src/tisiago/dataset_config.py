"""Declarative build config + provenance for the embedding dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    """Declarative parameters for one compact-store build."""

    name: str
    parts_dir: str
    out_store: str
    keys: list[str]
    neg_cap: int
    noncog_sample: int
    seed: int = 0
    shard_glob: str = "*_shard*.npz"  # selects shard partials, not output .npz in the same dir


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load a ``dataset.yaml`` into a ``DatasetConfig`` (unknown keys are ignored)."""
    d = yaml.safe_load(Path(path).read_text())
    return DatasetConfig(**{k: d[k] for k in DatasetConfig.__dataclass_fields__ if k in d})


def write_provenance(store: Path, cfg: DatasetConfig, keys_meta: dict, git_sha: str) -> None:
    """Write/merge the store's ``config.yaml`` provenance (accumulates ``keys`` across calls)."""
    cfg_path = store / "config.yaml"
    prov = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    prov.setdefault("keys", {})
    prov["keys"].update(keys_meta)
    prov["name"] = cfg.name
    prov["neg_cap"] = cfg.neg_cap
    prov["noncog_sample"] = cfg.noncog_sample
    prov["seed"] = cfg.seed
    prov["git_sha"] = git_sha
    with open(cfg_path, "w") as f:
        yaml.safe_dump(prov, f, default_flow_style=False, sort_keys=False)
