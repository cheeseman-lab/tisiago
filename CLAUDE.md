# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Project

**tisiago** v0.1.0 — the TIS-prediction orchestrator. The third repo in the
fry-python-tools translation-initiation pipeline:

```
swissisoform → manifest.parquet → tisiago (input construction) → gruyerenome forward → tisiago (store + heads)
 makes labels    candidates       genomic or spliced sequence     per-candidate vectors   training/eval
```

- **swissisoform** makes the table: `manifest.parquet`, one row per candidate
  translation-initiation codon (called TIS = positives + matched in-transcript
  negatives, with labels, expression flags, region class, chromosome split).
- **gruyerenome** owns foundation-model loading and forward passes: generic
  `embed_positions(sequences, positions)` over AlphaGenome / Evo2. No TIS knowledge. Do not
  duplicate a model implementation here.
- **tisiago** (this repo) is the task orchestrator: reads the table, constructs genomic windows
  or exon-spliced transcript inputs, asks gruyerenome for the hidden rows at each candidate,
  assembles a row-aligned vector store, and trains/evaluates lightweight downstream heads.

Design principle: **decouple the forward-pass input from what's persisted** — embed a generous
genomic window or the complete mature transcript, store only the candidate states. Evo2 is
autoregressive, but the installed Vortex intermediate states are not exactly suffix-invariant;
never assume padding or candidate-dependent truncation leaves a state unchanged. Run
`python -m tisiago.backend_contract` after any gruyerenome/model dependency change.

## Direction

The goal is a **general codon → P(initiation) predictor** that also confidently rejects
non-starts. Live status is in [`ROADMAP.md`](ROADMAP.md); results in [`FINDINGS.md`](FINDINGS.md).

- **Frozen-embedding line — done.** AlphaGenome + Evo2 embeddings with a logistic head, trained
  on dense in-transcript negatives at true imbalance: clean-protocol recall **0.306 @ 0.99
  FP/transcript**, AUPRC 0.304 (5 seeds, FINDINGS §10). Richer heads tie (§9). Recall is driven
  by AUG (0.58); near-cognate starts are mostly missed (0.07). This head is the **baseline**.
- **Next — P6, seq2func.** Fine-tune AlphaGenome to predict the raw Ribo-seq tracks (5 cell
  lines × 2 replicates) and derive start-site calls on top, instead of training a binary head on
  sparse labels. Needs a spec: per-line vs multi-output model, handling of untranscribed genes
  (missing data, not zero), where fine-tuning code lives.

**Evaluation rules** (`docs/INFERENCE_AND_EVALUATION.md`): group splits by transcript/chromosome;
keep train / calibration / operating-threshold / test disjoint; pick thresholds on validation,
never on test; judge against the one-hot ±20 bp sequence floor, not chance. Headline metrics:
recall at a validation-selected FP/transcript budget at true imbalance (stratified AUG vs
near-cognate), and non-cognate grounding ≈0.

## Pipeline & commands

Site paths, per-stage Python, and partitions come from the gitignored `.env` (keys in
`.env.example`). Every Slurm script sources `scripts/_common.sh`, which loads `.env` and writes
logs to `logs/`. Submit from the repo root.

```bash
# 1. table comes from swissisoform (separate repo) -> data/manifest.parquet
# 2. extract per-candidate embeddings (GPU) + assemble store
bash scripts/run_tis_pipeline.sh 20
sbatch --array=0-19%2 --partition=nvidia-A100-20 scripts/run_tis_extract_transcript.sh \
    configs/tis_evo2_8k_blk28.yaml 20          # exon-spliced Evo2 arm (evo2/TXP keys)
# 3. evaluate heads (CPU, tisiago env)
python -m tisiago.eval --store data/store --keys alphagenome_jax/L16k/decoder_1bp/off0.npy
sbatch scripts/run_tis_dense_representation_eval.sh   # clean multi-seed dense protocol (§10)
```

For interactive `python -m` calls that need site paths: `set -a; source .env; set +a`.

## Code Structure

```
src/tisiago/
  # input construction + extraction (GPU)
  tiling.py                grid-snap genomic windowing (nearby candidates share a forward pass)
  extract.py               genome fetch + gruyerenome embed_positions -> per-shard parts
  transcript_inference.py  pure construction of exon-spliced, transcript-oriented Evo2 requests
  extract_transcript.py    complete exon-spliced Evo2 extraction (evo2/TXP namespace)
  sequence.py              strand-aware DNA / transcript helpers
  backend_contract.py      sparse/batch/indexing semantic checks against gruyerenome
  compare_extractions.py   drift check between aligned shard artifacts across environments
  # stores
  extraction_io.py         memory-bounded, atomic shard output
  store.py                 assemble curated parts into row-aligned [N, D] .npy arrays
  shard_io.py              streaming shard reader (Regime A gather + Regime B scoring)
  dataset_config.py        declarative dense-store build config + provenance
  store_overlap.py         exact-identity check at sites shared by two stores
  manifest.py              manifest validation + unique-site selection
  enumerate_codons.py      GTF -> dense scan manifest (every codon)
  # heads + evaluation (CPU)
  eval.py, resolution.py   PoC: AUROC by start type; near-neighbour win-rate
  caller.py                calibration, reliability, recall @ FP/transcript budget
  dense_caller.py          curated vs dense-trained heads at true imbalance (--head switch)
  head_xgb.py              XGBoost / LightGBM / RF heads
  efficiency_head.py       Ridge regression on max_norm_HeLa (P4)
  representation_eval.py   clean multi-seed W8k vs TXP comparison with bootstrap CIs
  linear_head.py           portable calibrated linear-head artifact
  projected_score.py, transcript_projection.py   partial-logit scoring without full features
  scan_eval.py, scan_score.py                     dense-scan scoring (Regime B streaming)
scripts/        Slurm wrappers (_common.sh shared header), store builders, compare_heads, report
configs/        gruyerenome backend configs + dense dataset configs
autoresearch/   4-objective head-search harness (done; winners.md)
docs/           INFERENCE_AND_EVALUATION.md; superpowers/ specs + plans (historical record)
logs/           Slurm output (gitignored)
data/           manifest + stores (gitignored)
```

## Environments

Conda + uv, per stage (paths set in `.env`):

| Stage | Environment | Notes |
|---|---|---|
| Eval / training / assembly (CPU) | conda `tisiago` | `uv pip install -e ".[dev]"` |
| AlphaGenome extraction (GPU) | conda `alphagenome` | tisiago + gruyerenome editable, pyfaidx |
| Evo2 extraction (GPU) | uv `.venv/evo2-next` | built by `scripts/run_tis_evo2_env_setup.sh` from `requirements-evo2.lock` (Evo2 0.6 / Vortex 1.1, source-built FlashAttention) |

```bash
eval "$(conda shell.bash hook)" && conda activate tisiago
python -m pytest
```

Never commit `.env` or gated-model credentials.

## Store layout

Row-aligned to `manifest.row_idx` (row i = candidate with row_idx == i):

```
data/store/
  manifest.parquet
  config.yaml                                   # provenance: dims, coverage
  embeddings/
    alphagenome_jax/L16k/decoder_1bp/off0.npy   # [N, 1536] fp16
    alphagenome_jax/L131k/decoder_1bp/off0.npy
    evo2/W8k/blocks.28.mlp.l3/off{0,3,6,9}.npy  # [N, 4096] fp16, genomic 8 kb windows
    evo2/TXP/blocks.28.mlp.l3/off{0,3,6,9}.npy  # exon-spliced transcript input
    onehot/{codon12,kozakW20}.npy               # sequence floors
```

A head experiment = load a few `.npy`, `np.concatenate(axis=1)`, filter rows by `split`.

Dense data (~62.7 M codons) is never assembled into one matrix. Regime A
(`scripts/build_store.py`) gathers candidate rows into a compact store — `data/dense_exp_store`
(4.33 M rows; SSD twin at `$TISIAGO_DENSE_STORE`). Regime B (`tisiago.scan_score`) streams shards
to score every codon. See `docs/superpowers/specs/2026-06-18-embedding-dataset-architecture-decision.md`.

## Code Style

- Build: hatchling · Linter: ruff (line-length 100) · Docstrings: Google · Python >=3.11
- Library code (`src/`, `configs/`) must not embed site paths — `tests/test_portability.py`
  enforces it; site values belong in `.env`.
