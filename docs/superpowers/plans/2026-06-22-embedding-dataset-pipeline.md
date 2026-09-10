# Reproducible Embedding-Dataset Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GLM-embedding → trainable-dataset → genome-wide-eval pipeline reproducible and fast by replacing the "materialize a row-aligned monolith" architecture with (A) a config-driven compact gather onto SSD and (B) a streaming genome-wide scorer that never assembles a monolith.

**Architecture:** A single declarative `dataset.yaml` drives every build. Two shard-consuming code paths share one streaming reader: **Regime A** gathers only candidate rows into a compact memmap-per-key store on SSD (already prototyped in `gather_keys_from_shards.py`); **Regime B** streams every shard once, applies a calibrated head, writes per-shard scalar predictions, and aggregates — the full-`[N,D]` monolith assembly in `store.py` is retired for the dense scan. Decisions and citations: `docs/superpowers/specs/2026-06-18-embedding-dataset-architecture-decision.md` (Decided blocks A1–A4) + `2026-06-18-part3-literature-review.md`.

**Tech Stack:** Python ≥3.11, numpy, pandas, pyyaml, pytest, sklearn (existing head code). CPU-only. SLURM partition `20` for big-data runs.

## Global Constraints

- Python ≥3.11, type annotations on all public functions, Google-style docstrings.
- ruff line-length 100. Match each file's existing comment density (no new comment blocks where the file has none).
- pytest; `testpaths = ["tests"]`. New tests live in `tests/` mirroring the module name.
- Temp files in the working directory only — never `/tmp`. Use `tmp_path` (pytest) or `dir="."`.
- SSD store root: configurable site-local scratch storage.
- Memory rule: build matrices by **preallocation**, never parts-list + `np.concatenate` (holds 2× peak); cgroup RSS counts mmap page-cache, so size jobs as `arrays + working-set`.
- SLURM hygiene: `${USER}`-namespaced job names; **`scancel` by job ID only** (never by name).
- Conda: `eval "$(conda shell.bash hook)" && conda activate tisiago` before any python.
- Shard member naming is `backend::ltag::layer::off`; `row_idx` member = source coordinate (0..N_scan−1). Compact manifests carry `src_row_idx` (= shard coordinate) and contiguous `row_idx` (0..M−1).

---

### Task 1: `dataset.yaml` schema + provenance writer

**Files:**
- Create: `src/tisiago/dataset_config.py`
- Test: `tests/test_dataset_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_dataset_config(path: str | Path) -> DatasetConfig` — dataclass with fields
    `name: str`, `parts_dir: str`, `out_store: str`, `keys: list[str]`, `neg_cap: int`,
    `noncog_sample: int`, `seed: int`.
  - `write_provenance(store: Path, cfg: DatasetConfig, keys_meta: dict, git_sha: str) -> None` —
    writes `store/config.yaml` merging any prior `keys` block (per-key accumulation).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_config.py
from pathlib import Path
import yaml
from tisiago.dataset_config import load_dataset_config, write_provenance, DatasetConfig


def test_load_dataset_config_parses_required_fields(tmp_path):
    p = tmp_path / "dataset.yaml"
    p.write_text(yaml.safe_dump({
        "name": "dense_v1", "parts_dir": "data/scan_parts_allsplits",
        "out_store": "/ssd/tisiago_store/dense", "keys": ["evo2/W8k/blocks.28.mlp.l3/off0.npy"],
        "neg_cap": 1000000, "noncog_sample": 1000000, "seed": 0,
    }))
    cfg = load_dataset_config(p)
    assert isinstance(cfg, DatasetConfig)
    assert cfg.name == "dense_v1" and cfg.neg_cap == 1_000_000
    assert cfg.keys == ["evo2/W8k/blocks.28.mlp.l3/off0.npy"]


def test_write_provenance_merges_prior_keys(tmp_path):
    store = tmp_path / "store"; store.mkdir()
    cfg = DatasetConfig(name="d", parts_dir="p", out_store=str(store),
                        keys=["a.npy"], neg_cap=1, noncog_sample=1, seed=0)
    write_provenance(store, cfg, {"a::b::c::off0": {"dim": 4096, "covered": 10}}, "sha1")
    write_provenance(store, cfg, {"x::y::z::off0": {"dim": 1536, "covered": 10}}, "sha1")
    cfg_out = yaml.safe_load((store / "config.yaml").read_text())
    assert set(cfg_out["keys"]) == {"a::b::c::off0", "x::y::z::off0"}  # accumulated
    assert cfg_out["git_sha"] == "sha1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `eval "$(conda shell.bash hook)" && conda activate tisiago && pytest tests/test_dataset_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tisiago.dataset_config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tisiago/dataset_config.py
"""Declarative build config + provenance for the embedding dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    name: str
    parts_dir: str
    out_store: str
    keys: list[str]
    neg_cap: int
    noncog_sample: int
    seed: int = 0


def load_dataset_config(path: str | Path) -> DatasetConfig:
    d = yaml.safe_load(Path(path).read_text())
    return DatasetConfig(**{k: d[k] for k in DatasetConfig.__dataclass_fields__ if k in d})


def write_provenance(store: Path, cfg: DatasetConfig, keys_meta: dict, git_sha: str) -> None:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tisiago/dataset_config.py tests/test_dataset_config.py
git commit -m "feat(dataset): declarative dataset.yaml config + provenance writer"
```

---

### Task 2: Shared streaming shard reader (factor out of `gather_keys_from_shards.py`)

**Files:**
- Create: `src/tisiago/shard_io.py`
- Modify: `scripts/gather_keys_from_shards.py` (replace inline loop with the shared reader)
- Test: `tests/test_shard_io.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `build_src_to_compact(manifest: pd.DataFrame) -> np.ndarray` — int64 map of length
    `max(src_row_idx)+1`, value = compact `row_idx` or −1 if not wanted. Requires columns
    `src_row_idx`, `row_idx` (contiguous 0..M−1).
  - `iter_shards(parts_dir: Path, glob: str)` — generator yielding `(rows: np.ndarray, members: dict[str, np.ndarray])` per shard, where `members` excludes `row_idx`.
  - `map_rows(rows: np.ndarray, src2cmp: np.ndarray) -> tuple[np.ndarray, np.ndarray]` —
    returns `(keep_mask, dst)` where `dst = src2cmp[rows][keep_mask]`; masks indices ≥ map length (definitionally not-wanted).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shard_io.py
import numpy as np, pandas as pd
from pathlib import Path
from tisiago.shard_io import build_src_to_compact, map_rows, iter_shards


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shard_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tisiago.shard_io'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tisiago/shard_io.py
"""Shared streaming reader over extraction shards (Regime A gather + Regime B scoring)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def build_src_to_compact(manifest: pd.DataFrame) -> np.ndarray:
    src = manifest.src_row_idx.values
    src2cmp = np.full(int(src.max()) + 1, -1, dtype=np.int64)
    src2cmp[src] = manifest.row_idx.values
    return src2cmp


def map_rows(rows: np.ndarray, src2cmp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cmp = np.full(len(rows), -1, dtype=np.int64)
    in_range = rows < len(src2cmp)
    cmp[in_range] = src2cmp[rows[in_range]]
    keep = cmp >= 0
    return keep, cmp[keep]


def iter_shards(parts_dir: Path, glob: str) -> Iterator[tuple[np.ndarray, dict]]:
    for p in sorted(Path(parts_dir).glob(glob)):
        z = np.load(p)
        members = {k: z[k] for k in z.files if k != "row_idx"}
        yield z["row_idx"], members
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_shard_io.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Refactor `gather_keys_from_shards.py` to use the shared reader**

Replace the inline shard loop (the `for i, p in enumerate(parts)` block and the manual
`src2cmp`/mask construction) with:

```python
from tisiago.shard_io import build_src_to_compact, iter_shards, map_rows
...
    src2cmp = build_src_to_compact(manifest)
    arrays, covered = {}, {}
    for i, (rows, members) in enumerate(iter_shards(Path(args.parts_dir), args.glob)):
        keep, dst = map_rows(rows, src2cmp)
        for key, mat in members.items():
            if key not in arrays:
                arrays[key] = np.zeros((m, mat.shape[1]), dtype=np.float16)
                covered[key] = np.zeros(m, dtype=bool)
            arrays[key][dst] = mat[keep]
            covered[key][dst] = True
        print(f"  shard {i}: {int(keep.sum()):,} wanted rows", flush=True)
```

- [ ] **Step 6: Verify the refactor is bit-identical on a real shard**

Run:
```bash
cd /path/to/tisiago
python -c "
import numpy as np, pandas as pd
from pathlib import Path
from tisiago.shard_io import build_src_to_compact, map_rows
m = pd.read_parquet('data/dense_exp_store/manifest.parquet')
s2c = build_src_to_compact(m)
z = np.load('data/scan_parts_allsplits/evo2_8k_shard00038.npz')
keep, dst = map_rows(z['row_idx'], s2c)
assert keep.sum() == 45736, keep.sum()   # value confirmed in session 2026-06-20
print('OK', int(keep.sum()))
"
```
Expected: `OK 45736`.

- [ ] **Step 7: Commit**

```bash
git add src/tisiago/shard_io.py tests/test_shard_io.py scripts/gather_keys_from_shards.py
git commit -m "refactor(shard_io): shared streaming reader; gather uses it"
```

---

### Task 3: Regime B streaming scorer (retire the monolith dependency in `scan_eval.py`)

**Files:**
- Create: `src/tisiago/scan_score.py`
- Test: `tests/test_scan_score.py`

**Interfaces:**
- Consumes: `iter_shards`, `map_rows`, `build_src_to_compact` (Task 2); a fitted head dict with
  a `"predict"` callable returning P (from `caller.fit_calibrated_head`).
- Produces:
  - `score_shards(parts_dir: Path, glob: str, manifest: pd.DataFrame, keys: list[str], head: dict, chunk: int = 100_000) -> np.ndarray` — returns `p` aligned to `manifest.row_idx`
    (0..M−1), scoring every wanted position by streaming shards once. Concatenates the requested
    `keys` per shard (preallocated, fp32) and applies `head["predict"]`. Asserts full coverage.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scan_score.py
import numpy as np, pandas as pd
from pathlib import Path
from tisiago.scan_score import score_shards


def test_score_shards_aligns_to_manifest_via_streaming(tmp_path):
    # 3 wanted compact rows mapped from src {2,5,9}; one 2-col key
    key = "ag::L16k::decoder_1bp::off0"
    manifest = pd.DataFrame({"src_row_idx": [2, 5, 9], "row_idx": [0, 1, 2]})
    # shard A holds src 2 and 9, shard B holds src 5 (plus unwanted src 7)
    np.savez(tmp_path / "a.npz", row_idx=np.array([2, 9]),
             **{key: np.array([[1.0, 0.0], [3.0, 0.0]], np.float16)})
    np.savez(tmp_path / "b.npz", row_idx=np.array([5, 7]),
             **{key: np.array([[2.0, 0.0], [9.0, 0.0]], np.float16)})
    # head: P = first feature column (so we can assert ordering)
    head = {"predict": lambda X: X[:, 0]}
    p = score_shards(tmp_path, "*.npz", manifest, [key + ".npy"], head)
    # compact row 0<-src2=1.0, row1<-src5=2.0, row2<-src9=3.0
    assert p.tolist() == [1.0, 2.0, 3.0]


def test_score_shards_raises_on_incomplete_coverage(tmp_path):
    key = "ag::L16k::decoder_1bp::off0"
    manifest = pd.DataFrame({"src_row_idx": [2, 5, 9], "row_idx": [0, 1, 2]})
    np.savez(tmp_path / "a.npz", row_idx=np.array([2, 9]),
             **{key: np.ones((2, 2), np.float16)})  # src 5 never appears
    head = {"predict": lambda X: X[:, 0]}
    try:
        score_shards(tmp_path, "*.npz", manifest, [key + ".npy"], head)
        assert False, "expected coverage assertion"
    except AssertionError as e:
        assert "covered" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scan_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tisiago.scan_score'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tisiago/scan_score.py
"""Regime B: score every wanted position by streaming shards — no monolith assembly.

Streams each shard once, concatenates the requested feature keys for its wanted rows into a
preallocated fp32 block, applies the calibrated head, and scatters the scalar predictions into
a result aligned to ``manifest.row_idx``. The full ``[N_scan, D]`` store is never materialized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.shard_io import build_src_to_compact, iter_shards, map_rows


def score_shards(parts_dir: Path, glob: str, manifest: pd.DataFrame, keys: list[str],
                 head: dict, chunk: int = 100_000) -> np.ndarray:
    # member names carry "::" and no ".npy"; manifest keys are slash paths with ".npy"
    want = [k[:-4].replace("/", "::") if k.endswith(".npy") else k.replace("/", "::") for k in keys]
    m = len(manifest)
    src2cmp = build_src_to_compact(manifest)
    p = np.full(m, np.nan, dtype=np.float64)
    covered = np.zeros(m, dtype=bool)
    for rows, members in iter_shards(Path(parts_dir), glob):
        keep, dst = map_rows(rows, src2cmp)
        if not keep.any():
            continue
        widths = [members[w].shape[1] for w in want]
        X = np.empty((int(keep.sum()), sum(widths)), dtype=np.float32)
        c = 0
        for w, wid in zip(want, widths):
            X[:, c:c + wid] = members[w][keep]
            c += wid
        p[dst] = head["predict"](X)
        covered[dst] = True
    n_cov = int(covered.sum())
    assert n_cov == m, f"covered {n_cov}/{m} rows — shards do not cover all wanted positions"
    return p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scan_score.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Point `scan_eval.py` at the streaming scorer**

In `src/tisiago/scan_eval.py` `main()`, replace the monolith load + predict (the
`Xs = _load(...)` / `p = head["predict"](Xs)` lines, ~74–75) with:

```python
from tisiago.scan_score import score_shards
...
    sm = pd.read_parquet(scan / "manifest.parquet")
    p = score_shards(Path(args.parts_dir), args.glob, sm, args.keys, head)
```

Add the two args to the parser (near the existing `--scan-store`):

```python
    ap.add_argument("--parts-dir", default="data/scan_parts_allsplits")
    ap.add_argument("--glob", default="*.npz")
```

- [ ] **Step 6: Run the full test suite (no regressions)**

Run: `pytest tests/ -v`
Expected: PASS — `test_scan_eval.py` (grounding_stats) still green; new tests green.

- [ ] **Step 7: Commit**

```bash
git add src/tisiago/scan_score.py tests/test_scan_score.py src/tisiago/scan_eval.py
git commit -m "feat(scan_score): stream-and-score genome-wide; retire monolith in scan_eval"
```

---

### Task 4: Config-driven compact-store build onto SSD

**Files:**
- Create: `scripts/build_store.py`
- Create: `configs/dataset_dense_ag7.yaml`
- Test: `tests/test_build_store.py`

**Interfaces:**
- Consumes: `load_dataset_config`, `write_provenance` (Task 1); `gather_keys_from_shards` logic
  (Task 2 reader). Reuses the existing manifest in the out-store if present.
- Produces: `build_store(cfg: DatasetConfig, git_sha: str) -> Path` — gathers `cfg.keys` from
  `cfg.parts_dir` into `cfg.out_store` (memmap-per-key fp16), writes provenance, asserts full
  coverage per key. Returns the store path.

- [ ] **Step 1: Write the failing test (synthetic end-to-end)**

```python
# tests/test_build_store.py
import numpy as np, pandas as pd, yaml
from pathlib import Path
from tisiago.dataset_config import DatasetConfig
from scripts.build_store import build_store


def test_build_store_gathers_to_compact_with_provenance(tmp_path):
    parts = tmp_path / "parts"; parts.mkdir()
    out = tmp_path / "store"; out.mkdir()
    key = "ag::L16k::decoder_1bp::off0"
    # manifest: 3 wanted compact rows from src {2,5,9}
    pd.DataFrame({"src_row_idx": [2, 5, 9], "row_idx": [0, 1, 2]}).to_parquet(out / "manifest.parquet")
    np.savez(parts / "s0.npz", row_idx=np.array([2, 5, 9]),
             **{key: np.arange(6, dtype=np.float16).reshape(3, 2)})
    cfg = DatasetConfig(name="t", parts_dir=str(parts), out_store=str(out),
                        keys=["ag/L16k/decoder_1bp/off0.npy"], neg_cap=1, noncog_sample=1, seed=0)
    build_store(cfg, "deadbeef")
    arr = np.load(out / "embeddings/ag/L16k/decoder_1bp/off0.npy")
    assert arr.shape == (3, 2) and arr[2, 0] == 4  # src9 -> compact row 2
    prov = yaml.safe_load((out / "config.yaml").read_text())
    assert prov["git_sha"] == "deadbeef" and prov["keys"][key]["covered"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_store'` (or import error).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/build_store.py
"""Build a compact memmap-per-key store from shards, driven by a dataset.yaml."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.dataset_config import DatasetConfig, load_dataset_config, write_provenance
from tisiago.shard_io import build_src_to_compact, iter_shards, map_rows


def build_store(cfg: DatasetConfig, git_sha: str) -> Path:
    out = Path(cfg.out_store)
    manifest = pd.read_parquet(out / "manifest.parquet")
    m = len(manifest)
    assert (manifest.row_idx.values == np.arange(m)).all(), "row_idx must be 0..M-1"
    src2cmp = build_src_to_compact(manifest)
    want = [k[:-4].replace("/", "::") for k in cfg.keys]
    arrays, covered = {}, {}
    for rows, members in iter_shards(Path(cfg.parts_dir), "*.npz"):
        keep, dst = map_rows(rows, src2cmp)
        for w in want:
            if w not in members:
                continue
            mat = members[w]
            if w not in arrays:
                arrays[w] = np.zeros((m, mat.shape[1]), dtype=np.float16)
                covered[w] = np.zeros(m, dtype=bool)
            arrays[w][dst] = mat[keep]
            covered[w][dst] = True
    keys_meta = {}
    for w, mat in arrays.items():
        backend, ltag, layer, off = w.split("::")
        op = out / "embeddings" / backend / ltag / layer / f"{off}.npy"
        op.parent.mkdir(parents=True, exist_ok=True)
        np.save(op, mat)
        cov = int(covered[w].sum())
        assert cov == m, f"{w}: {cov}/{m} covered"
        keys_meta[w] = {"path": str(op.relative_to(out)), "dim": int(mat.shape[1]), "covered": cov}
    write_provenance(out, cfg, keys_meta, git_sha)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = build_store(load_dataset_config(args.config), sha)
    print(f"built store at {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_store.py -v`
Expected: PASS (1 passed). (If `scripts` is not importable, add `tests/conftest.py` with
`import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))`.)

- [ ] **Step 5: Author the real dense `ag7` config**

```yaml
# configs/dataset_dense_ag7.yaml
name: dense_ag7
parts_dir: data/scan_parts_allsplits
out_store: /path/to/dense_ag7
keys:
  - alphagenome_jax/L16k/decoder_1bp/off0.npy
  - alphagenome_jax/L131k/decoder_1bp/off0.npy
  - evo2/W8k/blocks.28.mlp.l3/off0.npy
  - evo2/W8k/blocks.28.mlp.l3/off3.npy
  - evo2/W8k/blocks.28.mlp.l3/off6.npy
  - evo2/W8k/blocks.28.mlp.l3/off9.npy
  - onehot/kozakW20.npy
neg_cap: 1000000
noncog_sample: 1000000
seed: 0
```

- [ ] **Step 6: Integration run (real data → SSD)**

The out-store manifest is reused from `data/dense_exp_store` (already built). Copy it into the
SSD out-store, then build via a big-mem SLURM job (sequential reads; ~hours on the 1.5 TB
Evo2 shards):

```bash
mkdir -p /path/to/dense_ag7
cp data/dense_exp_store/manifest.parquet /path/to/dense_ag7/
sbatch --job-name=${USER}_build_ag7 --partition=20 --cpus-per-task=8 --mem=240G \
  --time=18:00:00 --output=build_ag7.log \
  --wrap='eval "$(conda shell.bash hook)" && conda activate tisiago && \
          cd /path/to/tisiago && \
          python scripts/build_store.py --config configs/dataset_dense_ag7.yaml'
```
Validate: `grep -c covered build_ag7.log` shows each key `covered=4329984/4329984`; abort/raise
otherwise (the build asserts this). If a job is stuck pending, `scancel <jobid>` (by ID).

- [ ] **Step 7: Commit**

```bash
git add scripts/build_store.py configs/dataset_dense_ag7.yaml tests/test_build_store.py
git commit -m "feat(build_store): config-driven compact store onto SSD with provenance"
```

---

### Task 5: Retire the monolith assembly path + update docs

**Files:**
- Modify: `src/tisiago/store.py` (deprecate the dense full-`[N,D]` scatter)
- Modify: `ARCHITECTURE.md`, `CLAUDE.md` (store-layout sections)
- Test: `tests/test_store_deprecation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `store.py` emits a deprecation warning steering dense-scan callers to
  `scripts/build_store.py` (Regime A) or `scan_score.py` (Regime B); curated-store assembly
  (small, candidate-only) remains supported.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_deprecation.py
import warnings
import tisiago.store as store


def test_assemble_emits_deprecation_for_dense(monkeypatch, tmp_path):
    # the dense-scan monolith path should warn; pass a sentinel that triggers the guard
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store.warn_if_dense_monolith(n_rows=62_683_645)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert any("scan_score" in str(x.message) for x in w)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store_deprecation.py -v`
Expected: FAIL with `AttributeError: module 'tisiago.store' has no attribute 'warn_if_dense_monolith'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/tisiago/store.py` (above `main`):

```python
import warnings

DENSE_MONOLITH_THRESHOLD = 10_000_000  # candidate stores are ≪ this; dense scans are ≫


def warn_if_dense_monolith(n_rows: int) -> None:
    """Steer genome-scale assembly to the streaming/gather paths (decision doc A2)."""
    if n_rows >= DENSE_MONOLITH_THRESHOLD:
        warnings.warn(
            f"Assembling a {n_rows:,}-row monolith is retired for dense scans — use "
            "scripts/build_store.py (compact gather) or tisiago.scan_score (streaming eval).",
            DeprecationWarning,
            stacklevel=2,
        )
```

Then call it in `main()` right after `n = len(manifest)`:

```python
    warn_if_dense_monolith(n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store_deprecation.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Update the store-layout docs**

In `ARCHITECTURE.md` and `CLAUDE.md`, in the store-layout section, add a line stating: dense
genome-wide data is **not** assembled into row-aligned monoliths — Regime A
(`scripts/build_store.py`) gathers candidate rows into a compact SSD store; Regime B
(`tisiago.scan_score`) streams shards to score every codon. Reference the decision doc.

- [ ] **Step 6: Run full suite + commit**

```bash
pytest tests/ -v
git add src/tisiago/store.py tests/test_store_deprecation.py ARCHITECTURE.md CLAUDE.md
git commit -m "refactor(store): deprecate dense monolith assembly; document A/B regimes"
```

---

## Self-Review

**Spec coverage** (decision doc Decided blocks):
- A1 sharding contract → partially (member-name/`src_row_idx` handling is codified in `shard_io`; a separate shard-index emitter is deferred — noted below).
- A2 gather-for-A / stream-for-B + retire monolith → Tasks 2, 3, 5. ✓
- A3 memmap-per-key on SSD → Task 4 (config `out_store` on SSD; fp16 `.npy` per key). ✓
- A4 dataset.yaml + provenance + coverage asserts → Tasks 1, 4. ✓
- D1/D2/D3 (caller science) → **out of scope for this plan** (data-layer plan). The Dense(None)
  training + isotonic-at-true-prior caller and the D1 negative control are separate work
  (tracked as their own tasks); `scan_score` provides the substrate they plug into.

**Deferred (call out, not silently dropped):** the A1 "declared shard-index + single N" emitter
is not built here — the current per-backend N (AG=60/Evo2=80) works because mapping is by
`src_row_idx`. Add a `dataset.yaml`-level shard manifest in a follow-up if cross-backend N
divergence ever causes confusion.

**Placeholder scan:** none — every code/test step contains complete content.

**Type consistency:** `build_src_to_compact`/`map_rows`/`iter_shards` signatures match across
Tasks 2–4; member-name normalization (`"/"`→`"::"`, strip `.npy`) is identical in `scan_score`
and `build_store`; head `"predict"` callable matches `caller.fit_calibrated_head`'s contract.
