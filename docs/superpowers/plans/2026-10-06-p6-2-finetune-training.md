# P6.2 — Fine-Tuning AlphaGenome on Ribo-seq Tracks: Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train fine-tuned AlphaGenome models on many genomic windows — one shared 36-track model and six per-condition 6-track models — and evaluate predicted tracks on held-out chr7 against the replicate ceiling.

**Architecture:** Pure-numpy data layer (`expression.py`, `windows.py`, `data.py`, `metrics.py`) tested in the CPU `tisiago` env; JAX training/eval (`train.py`, `evaluate.py`) in `.venv/agft`, reusing P6.1's `model.make_train_step` (backbone fine-tuning, warmup-cosine, clipping), `targets.load_targets`, `augment`, `masking`. Batches are built on the fly (pyfaidx + pyBigWig) in a background prefetch thread. Selection uses chr7 only; chr8/chr9 are never read by training or selection code.

**Tech Stack:** as P6.1 (`.venv/agft`: jax 0.10.2, alphagenome 0.9.0, alphagenome-ft 0.1.12) + numpy, pandas, pyBigWig, pyfaidx; Slurm GPU `nvidia-L40S-24` / `nvidia-A6000-20`.

**Spec:** `docs/superpowers/specs/2026-10-05-p6-seq2func-ribo-tracks-design.md` (§4 P6.2, §5, §7). Evidence: P6.1 ledger + `FINDINGS.md` §11 (frozen backbone cannot fit base-resolution profiles; backbone fine-tuning with warmup-cosine + clip 1.0 reaches profile KL 0.76 on one window vs replicate floor ≈ 0.6).

## Global Constraints

- **Primary arm is backbone fine-tuning** (user steer 2026-10-06): `train_backbone=True`, head lr 1e-3, backbone lr 1e-4, `schedule="cosine"`, `warmup_steps=300`, `clip_norm=1.0`, MLP `TrackHead`, batch 1, window 131,072 bp. Heads-only is reported only as a baseline row.
- **Arms:** `shared` = one model, conditions `CONDITIONS` (6) → 36 channels; `condition:<C>` = one model per condition → 6 channels. Same windows, same steps, same seeds.
- **Splits by chromosome:** val = `chr7`; test = `chr8`, `chr9`; train = all other primary chromosomes `chr1`–`chr22`, `chrX` (exclude `chrY`, `chrM`, unplaced). Training and model selection never read chr8/chr9.
- **Expression:** a gene is expressed in a condition when its summed TIS-input RNA counts over both replicates (htseq files `/lab/barcheese01/aTIS_data/counts/{sample_id}_htseqcount.txt`, `sample_id` of the condition's `rna` library) are ≥ 2 — this reproduces the manifest's `expressed_*` flags exactly. Gene spans from the GENCODE v49 GTF `gene` lines (0-based half-open), gene_id with version, strand.
- **Windows:** union of spans of genes expressed in ≥ 1 condition, merged per chromosome, tiled with 131,072-bp windows at stride 65,536 covering every merged span; window start ≥ 0 and end ≤ chromosome length (shift inward when needed).
- **Targets/mask/augment:** `load_targets(..., rep="pooled")` canonical P-sites; `expression_mask` per condition; training augmentation = reverse-complement with p = 0.5 then a uniform integer shift in [-3, 3].
- **Metrics:** profile KL (deviance form, per channel, masked, averaged over windows); gene-level Pearson of log1p totals over expressed-gene spans per channel; both also computed rep1-vs-rep2 as the ceiling. Selection metric = mean val profile KL over channels.
- **Compute courtesy:** ≤ 2 concurrent GPU jobs; job names `mdiberna_*`; logs in `logs/`; checkpoints under `data/p6/runs/<arm>/` (gitignored), keep only `best/` and `last/`.
- ruff line length 100, Google docstrings; never `/tmp`.

## Review Focus

- A window must never straddle train and val/test chromosomes, and chr8/chr9 windows must be unreachable from the training iterator and the selection metric (Task 2, Task 5).
- The prefetch thread must not deadlock or swallow exceptions: a failing bigWig read in the worker must surface in the main thread (Task 3).
- RC augmentation must be applied identically to sequence, targets and mask, using `strand_pairs` of the arm's own layout (Task 3).
- Gene-level Pearson with < 3 expressed genes in a window set must return NaN, not crash; KL must ignore fully masked channels (Task 4).
- A crash mid-training must leave a usable `last/` checkpoint and a JSON log with all completed eval points (Task 5).

---

### Task 1: Per-condition expressed-gene table

**Files:** Create `src/tisiago/seq2func/expression.py`; Test `tests/test_seq2func_expression.py`

**Interfaces:**
- Consumes: `samples.load_samples(atis_root)` (rows with `library == "rna"`, `sample_id`).
- Produces: `read_gene_spans(gtf) -> pd.DataFrame[gene_id, chrom, start, end, strand]` (GTF `gene` lines; start = col4 − 1, end = col5); `read_counts(path) -> pd.Series` (index gene_id, int; lines starting `__` dropped); `expression_table(atis_root, gtf, min_count=2) -> pd.DataFrame[condition, gene_id, chrom, start, end, strand, rna_count, expressed]` (one row per condition × gene present in the GTF; counts summed over the two replicates' rna samples; missing gene in counts = 0).

- [ ] **Step 1: Failing tests**

```python
"""Tests for the per-condition expressed-gene table."""

import pandas as pd

from tisiago.seq2func.expression import expression_table, read_counts, read_gene_spans


def _gtf(tmp_path):
    path = tmp_path / "g.gtf"
    path.write_text(
        "chr1\tHAVANA\tgene\t11\t20\t.\t+\t.\tgene_id \"G1.1\"; gene_type \"x\";\n"
        "chr1\tHAVANA\texon\t11\t20\t.\t+\t.\tgene_id \"G1.1\";\n"
        "chr2\tHAVANA\tgene\t5\t9\t.\t-\t.\tgene_id \"G2.3\";\n"
    )
    return path


def test_read_gene_spans_zero_based(tmp_path):
    spans = read_gene_spans(_gtf(tmp_path))
    assert spans.to_dict("records") == [
        {"gene_id": "G1.1", "chrom": "chr1", "start": 10, "end": 20, "strand": "+"},
        {"gene_id": "G2.3", "chrom": "chr2", "start": 4, "end": 9, "strand": "-"},
    ]


def test_read_counts_drops_htseq_summary(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("G1.1\t3\nG2.3\t0\n__no_feature\t99\n")
    assert read_counts(path).to_dict() == {"G1.1": 3, "G2.3": 0}


def test_expression_table_sums_replicates_and_thresholds(tmp_path, monkeypatch):
    samples = pd.DataFrame(
        [
            {"condition": "HeLa", "library": "rna", "rep": 1, "sample_id": "a"},
            {"condition": "HeLa", "library": "rna", "rep": 2, "sample_id": "b"},
            {"condition": "HeLa", "library": "tis", "rep": 1, "sample_id": "t"},
        ]
    )
    monkeypatch.setattr("tisiago.seq2func.expression.load_samples", lambda root: samples)
    counts = tmp_path / "counts"
    counts.mkdir()
    (counts / "a_htseqcount.txt").write_text("G1.1\t1\nG2.3\t0\n")
    (counts / "b_htseqcount.txt").write_text("G1.1\t1\n")
    table = expression_table(tmp_path, _gtf(tmp_path))
    row = table.set_index("gene_id")
    assert row.loc["G1.1", "rna_count"] == 2 and bool(row.loc["G1.1", "expressed"])
    assert row.loc["G2.3", "rna_count"] == 0 and not bool(row.loc["G2.3", "expressed"])
    assert set(table.condition) == {"HeLa"}
```

- [ ] **Step 2: Run to verify failure** (`tisiago` env).
- [ ] **Step 3: Implement**

```python
"""Per-condition expressed-gene table from TIS-input RNA counts (manifest-equivalent rule)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tisiago.seq2func.samples import load_samples

_GENE_ID = re.compile(r'gene_id "([^"]+)"')


def read_gene_spans(gtf: str | Path) -> pd.DataFrame:
    """GTF ``gene`` lines as 0-based half-open spans."""
    rows = []
    with open(gtf) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            rows.append(
                {
                    "gene_id": _GENE_ID.search(f[8]).group(1),
                    "chrom": f[0],
                    "start": int(f[3]) - 1,
                    "end": int(f[4]),
                    "strand": f[6],
                }
            )
    return pd.DataFrame(rows, columns=["gene_id", "chrom", "start", "end", "strand"])


def read_counts(path: str | Path) -> pd.Series:
    """htseq-count gene counts, summary lines (``__*``) dropped."""
    table = pd.read_csv(path, sep="\t", header=None, names=["gene_id", "count"])
    table = table[~table.gene_id.str.startswith("__")]
    return table.set_index("gene_id")["count"].astype(int)


def expression_table(atis_root: str | Path, gtf: str | Path, min_count: int = 2) -> pd.DataFrame:
    """One row per condition x gene: summed TIS-input counts and ``expressed`` flag."""
    root = Path(atis_root)
    spans = read_gene_spans(gtf)
    samples = load_samples(root)
    frames = []
    for condition, rows in samples[samples.library == "rna"].groupby("condition"):
        total = pd.Series(0, index=spans.gene_id, dtype=int)
        for sample_id in rows.sample_id:
            counts = read_counts(root / "counts" / f"{sample_id}_htseqcount.txt")
            total = total.add(counts.reindex(spans.gene_id, fill_value=0), fill_value=0)
        frame = spans.copy()
        frame.insert(0, "condition", condition)
        frame["rna_count"] = total.to_numpy().astype(int)
        frame["expressed"] = frame.rna_count >= min_count
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Tests pass; real-data check** — `expression_table("/lab/barcheese01/aTIS_data", "<GTF>")` on a CPU node (`srun --partition=20 --mem=16G`): agreement of `expressed` with the manifest `expressed_{C}` flags over manifest genes must be 1.0 for every condition (print per-condition agreement; any < 1.0 is a finding, report it). Save `data/p6/expression/expression.parquet`.
- [ ] **Step 5: Commit** — `git commit -m "feat(p6): per-condition expressed-gene table"`

---

### Task 2: Training/validation windows

**Files:** Create `src/tisiago/seq2func/windows.py`; Test `tests/test_seq2func_windows.py`

**Interfaces:**
- Produces: `TRAIN_CHROMS`, `VAL_CHROMS = ("chr7",)`, `TEST_CHROMS = ("chr8", "chr9")`; `merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]`; `tile(spans, chrom_size, length=131072, stride=65536) -> list[int]` (window starts covering every merged span, clamped to `[0, chrom_size - length]`, deduplicated, sorted); `make_windows(expression: pd.DataFrame, chrom_sizes: dict[str, int], split: str) -> pd.DataFrame[chrom, start, end]` for `split in {"train", "val", "test"}` using genes expressed in ≥ 1 condition.

- [ ] **Step 1: Failing tests**

```python
"""Tests for chromosome-split window tiling."""

import pandas as pd
import pytest

from tisiago.seq2func.windows import make_windows, merge_spans, tile


def test_merge_spans():
    assert merge_spans([(5, 10), (0, 3), (8, 12), (12, 13)]) == [(0, 3), (5, 13)]


def test_tile_covers_and_clamps():
    starts = tile([(0, 10), (250, 260)], chrom_size=300, length=100, stride=50)
    assert starts == [0, 200]
    assert all(0 <= s <= 200 for s in starts)


def test_tile_long_span_uses_stride():
    assert tile([(0, 260)], chrom_size=1000, length=100, stride=50) == [0, 50, 100, 150, 200]


def test_make_windows_respects_split():
    expr = pd.DataFrame(
        [
            ("HeLa", "chr1", 0, 50, True),
            ("HeLa", "chr7", 0, 50, True),
            ("HeLa", "chr8", 0, 50, True),
            ("K562", "chr2", 0, 50, False),
        ],
        columns=["condition", "chrom", "start", "end", "expressed"],
    )
    sizes = {"chr1": 1000, "chr2": 1000, "chr7": 1000, "chr8": 1000}
    train = make_windows(expr, sizes, "train", length=100, stride=50)
    assert set(train.chrom) == {"chr1"}
    assert set(make_windows(expr, sizes, "val", length=100, stride=50).chrom) == {"chr7"}
    assert set(make_windows(expr, sizes, "test", length=100, stride=50).chrom) == {"chr8"}
    with pytest.raises(ValueError):
        make_windows(expr, sizes, "other")
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement**

```python
"""Chromosome-split 131 kb windows tiled over expressed genes."""

from __future__ import annotations

import pandas as pd

LENGTH = 131_072
STRIDE = 65_536
VAL_CHROMS = ("chr7",)
TEST_CHROMS = ("chr8", "chr9")
TRAIN_CHROMS = tuple(
    f"chr{c}" for c in [*range(1, 23), "X"] if f"chr{c}" not in VAL_CHROMS + TEST_CHROMS
)
SPLITS = {"train": TRAIN_CHROMS, "val": VAL_CHROMS, "test": TEST_CHROMS}


def merge_spans(spans):
    """Merge overlapping or touching half-open spans."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def tile(spans, chrom_size: int, length: int = LENGTH, stride: int = STRIDE) -> list[int]:
    """Window starts covering every span, clamped inside the chromosome."""
    last = max(chrom_size - length, 0)
    starts = set()
    for start, end in merge_spans(spans):
        position = start
        while True:
            starts.add(min(max(position, 0), last))
            if position + length >= end:
                break
            position += stride
    return sorted(starts)


def make_windows(
    expression: pd.DataFrame,
    chrom_sizes: dict[str, int],
    split: str,
    length: int = LENGTH,
    stride: int = STRIDE,
) -> pd.DataFrame:
    """Windows for one split from genes expressed in at least one condition."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    genes = expression[expression.expressed.astype(bool)]
    rows = []
    for chrom in SPLITS[split]:
        spans = genes[genes.chrom == chrom][["start", "end"]].itertuples(index=False)
        if chrom not in chrom_sizes:
            continue
        for start in tile(list(spans), chrom_sizes[chrom], length, stride):
            rows.append((chrom, start, start + length))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])
```

- [ ] **Step 4: Tests pass; real-data count** — build train/val/test windows from `data/p6/expression/expression.parquet` and the FASTA `.fai` sizes; print counts per split; save `data/p6/windows/{split}.parquet`.
- [ ] **Step 5: Commit** — `git commit -m "feat(p6): chromosome-split windows over expressed genes"`

---

### Task 3: Batch iterator with augmentation and prefetch

**Files:** Create `src/tisiago/seq2func/data.py`; Test `tests/test_seq2func_data.py`

**Interfaces:**
- Consumes: `load_targets`, `channel_layout`, `strand_pairs`, `expression_mask`, `reverse_complement`, `shift`.
- Produces: `one_hot(seq: str) -> np.ndarray (S, 4) float32` (ACGT; N/other → zeros); `make_example(fasta, tracks_dir, conditions, genes, chrom, start, end) -> dict(sequence, targets, mask)`; `augment(example, rng, pairs, max_shift=3) -> dict` (RC with p = 0.5, then shift in [-max_shift, max_shift]); `iterate(windows: pd.DataFrame, *, fasta, tracks_dir, conditions, genes, seed, augment_examples: bool, prefetch: int = 4, loop: bool = True) -> Iterator[dict]` yielding batch-1 dicts with a leading batch axis; a background thread fills a bounded queue; worker exceptions are re-raised in the consumer; `loop=False` yields each window once in order (for eval).

- [ ] **Step 1: Failing tests** — build a tiny FASTA (`chr1` 64 bp; write `.fai` via `pyfaidx.Fasta`) and tiny pooled tracks with `write_bigwigs` for one condition; assert: `one_hot("ACGTN")` rows; `make_example` shapes `(32, 4)`, `(32, 6)`, `(32, 6)` and a known canonical target value; `augment` with a seeded rng that triggers RC maps a plus count at x to minus at S−1−x (± shift); `iterate(..., loop=False)` yields exactly len(windows) batches with leading axis 1; a missing bigWig path makes `next(iterate(...))` raise in the main thread (not hang) — guard the test with a 30 s timeout via `pytest`'s `faulthandler_timeout` or a `threading.Timer`.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — `one_hot` via a 256-entry lookup table; `make_example` uses `pyfaidx.Fasta(fasta)[chrom][start:end]` (uppercase; pad with N outside the chromosome like `overfit.fetch_onehot`), `load_targets(tracks_dir, conditions, chrom, start, end)`, `expression_mask(channel_layout(conditions), genes, chrom, start, end)` where `genes` is the expression table filtered to `conditions`; `iterate` shuffles windows each epoch with `np.random.default_rng(seed + epoch)` when `loop=True`; worker thread puts examples (or a sentinel carrying the exception) into `queue.Queue(maxsize=prefetch)`; consumer re-raises; daemon thread.
- [ ] **Step 4: Tests pass; real-data timing** — on a CPU node, time 20 examples from `data/p6/windows/train.parquet` for the shared arm (36 channels) and one condition (6 channels); report seconds/example (must be < 0.26 s for the 6-channel arm to keep the GPU busy with prefetch; report the 36-channel number either way).
- [ ] **Step 5: Commit** — `git commit -m "feat(p6): augmented prefetching batch iterator"`

---

### Task 4: Track metrics

**Files:** Create `src/tisiago/seq2func/metrics.py`; Test `tests/test_seq2func_metrics.py`

**Interfaces:**
- Produces (numpy): `profile_kl(pred (S,C), target (S,C), mask (S,C)) -> np.ndarray (C,)` (deviance-form KL of target profile vs predicted profile over unmasked positions; NaN for a channel with zero unmasked target counts); `gene_totals(track (S,C), genes_in_window: pd.DataFrame, window_start) -> np.ndarray (G, C)` (sum over each gene span clipped to the window, strand-agnostic sum of the channel); `log_pearson(pred_totals (N,C), obs_totals (N,C)) -> np.ndarray (C,)` (Pearson of log1p per channel; NaN when < 3 rows or zero variance).

- [ ] **Step 1: Failing tests** — `profile_kl(y, y, mask)` == 0 per channel; a fully masked channel → NaN; proportional prediction (2·y) → 0 (profile is scale-free); `log_pearson` on perfectly proportional totals → 1.0; with 2 rows → NaN.
- [ ] **Step 2–3: Implement** with `np.where`/`np.log` guards mirroring `loss.poisson_multinomial`.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(p6): profile KL and gene-level log-count metrics"`

---

### Task 5: Training loop with validation selection and checkpoints

**Files:** Create `src/tisiago/seq2func/train.py`, `scripts/run_p6_train.sh`; Test `tests/test_seq2func_train.py`

**Interfaces:**
- Consumes: `model.register_track_head`, `load_model`, `make_train_step` (P6.1), `data.iterate`, `metrics.profile_kl`, `windows` parquet files.
- Produces: `train_loop(step_fn, params, state, opt_state, batches, *, steps, eval_every, eval_fn, on_improve, on_checkpoint, log_path) -> dict` — pure orchestration (no JAX imports) so it is CPU-testable with fakes: calls `step_fn` `steps` times, every `eval_every` steps calls `eval_fn(params)` → `{"val_profile_kl": float, ...}`, calls `on_improve(params, step)` when `val_profile_kl` improves, calls `on_checkpoint(params, opt_state, step)` every `eval_every` steps, appends one JSON line per eval to `log_path` (flushed), and returns the best record. CLI `python -m tisiago.seq2func.train --arm shared|condition:<C> --steps N --eval-every K --eval-windows M --out data/p6/runs/<arm>` that wires real objects: P6.1 defaults from Global Constraints; eval on the first `M` val windows (fixed, `loop=False`, no augmentation) computing mean masked `profile_kl` over channels; checkpoints via `model._params = params; model.save_checkpoint(out / "best" | "last", save_full_model=True)` (verify in `.venv/agft/lib/python3.11/site-packages/alphagenome_ft/custom_model.py:784` that it saves `self._params`; if not, save `params` with `orbax.checkpoint.StandardCheckpointer`).

- [ ] **Step 1: Failing tests** with fakes: `step_fn` decrements a counter; `eval_fn` returns a scripted sequence of KLs [3.0, 2.0, 2.5]; assert `on_improve` called at the 1st and 2nd evals only, `on_checkpoint` at every eval, log has 3 JSON lines with step numbers, returned best = 2.0 at its step; an exception raised by `step_fn` mid-run leaves the log with the completed eval lines (re-raise after flushing).
- [ ] **Step 2–3: Implement** `train_loop` + CLI + `scripts/run_p6_train.sh` (shared header; `--gres=gpu:1 --cpus-per-task=8 --mem=96G --time=7-00:00:00`; runtime env vars as P6.1 Global Constraints; `PYTHON="${PYTHON_BIN:-$TISIAGO_AGFT_PYTHON}"`).
- [ ] **Step 4: CPU tests pass; ruff; full suite.**
- [ ] **Step 5: Commit** — `git commit -m "feat(p6): training loop with chr7 selection and checkpoints"`
- [ ] **Step 6 (controller): smoke** — `--arm condition:HeLa --steps 600 --eval-every 200 --eval-windows 20` on L40S/A6000; check loss falls, eval KL is finite, `best/` and `last/` exist and reload with `alphagenome_ft.load_checkpoint`.

---

### Task 6: Held-out evaluation with replicate ceiling

**Files:** Create `src/tisiago/seq2func/evaluate.py`, `scripts/run_p6_evaluate.sh`; Test `tests/test_seq2func_evaluate.py`

**Interfaces:**
- Produces: `ceiling_metrics(tracks_dir, conditions, genes, windows) -> pd.DataFrame` (rep1 targets vs rep2 used as the "prediction": per-channel mean `profile_kl` and gene-level `log_pearson`) — CPU only; `evaluate_checkpoint(checkpoint, arm, windows, ...) -> pd.DataFrame` (same metrics for model predictions on pooled targets, plus CHX predicted frame-0 fraction over MANE CDS in the windows using `qc._frame_counts` semantics with canonical P-sites — i.e. no `RIBOTISH_MINUS_SHIFT`, since targets are canonical); CLI `python -m tisiago.seq2func.evaluate --checkpoint DIR --arm ARM --split val` writing `data/p6/eval/<arm>_<split>.tsv` and the ceiling to `data/p6/eval/ceiling_<split>.tsv`. `--split test` requires `--i-am-final` (refuses otherwise).

- [ ] **Step 1: Failing tests** — ceiling on identical rep1/rep2 tiny tracks gives KL 0 and Pearson 1; `--split test` without `--i-am-final` exits non-zero.
- [ ] **Step 2–4: Implement; tests pass; ruff; full suite.**
- [ ] **Step 5: Commit** — `git commit -m "feat(p6): held-out evaluation with replicate ceiling"`

---

### Task 7 (controller): Train the arms

- [ ] **Step 1:** compute the val ceiling (`evaluate.py` ceiling only, CPU).
- [ ] **Step 2:** train `shared` and the six `condition:<C>` arms (30,000 steps each, eval every 1,000 on 100 val windows), ≤ 2 GPU jobs at a time, shared arm first.
- [ ] **Step 3:** full chr7 evaluation of each arm's `best/`; heads-only baseline row = one `condition:HeLa` run with `--heads-only` for 30,000 steps (add the flag only if Task 5 exposes it; otherwise record "not run").
- [ ] **Step 4:** apply the track part of spec §7 (M2 = mean per-channel gene-level Pearson / ceiling); M1 (calls) belongs to P6.4. Record in the ledger.

### Task 8: Record P6.2

- [ ] `FINDINGS.md` §11 (training table: arm, steps, val profile KL, val gene Pearson, ceiling, ratio, GPU-hours) and `ROADMAP.md`; commit `docs(p6): P6.2 training results`.
