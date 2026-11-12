# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Project

**tisiago** — translation-initiation site (TIS) prediction from sequence and a cell line's RNA-seq.
Live work: fine-tuned AlphaGenome predicting base-resolution harringtonine (TIS) and CHX Ribo-seq
tracks, held out one cell line at a time, and start calls read from those tracks.

**Start here:** [`docs/PLAN.md`](docs/PLAN.md) (naming, status, objectives, phases, decisions).
Results in [`FINDINGS.md`](FINDINGS.md) (§ numbers are stable; issues and specs cite them).
What transfers to a new recipe: [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md). Label hand-off:
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).

### Names (docs/PLAN.md §1)

| Line | Package | What |
|---|---|---|
| tracks | `tisiago.tracks` | fine-tuned, RNA-conditioned AlphaGenome -> TIS + CHX tracks |
| calling | `tisiago.calling` | tracks -> start calls (caller-free score; native caller next) |
| truth | `tisiago.truth` | reproducible-start truth set + tagging + P-site check |
| evaluation | `tisiago.evaluation` | M1 / M2 start metrics |
| baseline / floor | `tisiago.baseline` | frozen AlphaGenome + logistic head; one-hot floors |

Tracks run IDs: `tracks.<stage>.<chx>.<cond>.ho-<line>.s<seed>` — stage `genomic|spliced`, chx
`free|flux`, cond `pos|film|corgi` (`@head|@trunk`), holdout `K562|HeLa|U2OS|RPE1`. Reference recipe:
`tracks.spliced.flux.pos`. Do not reintroduce P-numbers, step numbers or class numbers in code,
configs or new docs.

AlphaGenome only. Evo2 was removed 2026-11-12; the pre-cleanup tree is tag
`archive/pre-cleanup-2026-11-12`.

## Evaluation rules

`docs/INFERENCE_AND_EVALUATION.md`. Splits: val chr7, test chr8/9; leave-one-cell-line-out with RPE1's
three states as one group. Decide on val with 3 seeds (differences under ~1.5 points are noise); score test
once behind `--i-am-final`. Judge against `floor.*` and `baseline.ag`, never chance. Start metrics
against reproducible starts only, split AUG / AAG / AGG / other near-cognate and in-CDS vs outside — most
in-CDS in-frame near-cognate sites are elongation pauses (FINDINGS §17). Do not climb percent-of-replicate
or synthetic-BAM Jaccard (guardrails only).

## Code structure

```
src/tisiago/
  tracks/      samples, psite, track_io, targets, data, augment, masking, rna_features, expression,
               windows, qc                                  data: BAMs -> P-site tracks -> windows
               model (registered head: rnacond_fc2d), loss, head_kinds, checkpoint
               train_genomic     stage 1: genomic windows, genome-space head (--arm rnacond:<line>)
               txcache, cache_seq  transcript-space embedding cache (+ mRNA sequence)
               fastloop          head/loss/structure screening on the cache (~4 min/config)
               train_spliced     stage 2: backbone on spliced mRNA + transcript-space head
               predict_tracks, evaluate, periodicity, scorecard, metrics   prediction + M3 guardrails
  calling/     track_score (caller-free start score on the baseline protocol), ribotish (call parsing)
  truth/       schema, load, psite_check, differential
  evaluation/  start_metrics (M1, M2)
  baseline/    manifest, enumerate_codons, sequence, tiling, extract, extraction_io, store,
               shard_io, dataset_config, backend_contract, onehot, caller, linear_head,
               evaluate (multi-seed baseline vs floors), call_starts, scan_score
scripts/   run_tracks_*.sh, run_calling_*.sh, run_baseline_*.sh (Slurm; source _common.sh)
configs/   tracks/ (reference recipe), tracks/fastloop/, tis_alphagenome_*.yaml, dataset_dense_ag.yaml
docs/      PLAN, PRINCIPLES, INFERENCE_AND_EVALUATION, DATA_CONTRACT, WORKPLAN, specs/
```

Checkpoints register their head as `p6_rnacond_<line>`; keep that name so existing checkpoints load. The
per-condition arms and explored head kinds are gone (tag `archive/pre-cleanup-2026-11-12` has them).

## Environments

| Stage | Environment |
|---|---|
| CPU: baseline evaluation, truth, metrics, non-JAX tests | conda `tisiago` (`uv pip install -e ".[dev,extract,tracks]"`) |
| GPU: baseline extraction | conda `alphagenome` |
| GPU: tracks training/prediction; JAX tests | uv `.venv/agft` (`scripts/run_tracks_env_setup.sh`, `requirements-agft.lock`) |

```bash
eval "$(conda shell.bash hook)" && conda activate tisiago && python -m pytest     # CPU suite
srun -p 20 -c 8 --mem=32G -t 1:00:00 bash -c \
  'JAX_PLATFORMS=cpu PYTHONPATH=src .venv/agft/bin/python -m pytest tests/test_tracks_*.py \
   tests/test_calling_*.py --continue-on-collection-errors'   # JAX suite (no pysam there)
```

Site paths and partitions come from the gitignored `.env` (keys in `.env.example`); every Slurm script
sources `scripts/_common.sh`. For interactive `python -m` calls: `set -a; source .env; set +a`. Never
commit `.env` or model credentials.

## Code style

hatchling · ruff (line-length 100) · Google docstrings · Python >= 3.11. Library code (`src/`,
`configs/`) must not embed site paths — `tests/test_portability.py` enforces it.
