# tisiago

**Translation-initiation site (TIS) prediction from sequence and a cell line's RNA-seq.**

Given the RNA-seq of a cell line, predict where ribosomes initiate — base-resolution harringtonine
(initiation) and cycloheximide (elongation) Ribo-seq tracks — and turn those tracks into start calls that
hold up in a cell line the model never saw.

| Line | What | Package |
|---|---|---|
| **tracks** | AlphaGenome fine-tuned on Ribo-seq, conditioned on the line's RNA-seq, held out one cell line at a time. The live model. | `tisiago.tracks` |
| **calling** | Start calls from tracks: the caller-free score on the baseline's protocol, and (next) a native caller with a learned threshold. | `tisiago.calling` |
| **truth** | The reproducible-start truth set from the label pipeline, tagged by codon class, region and pause. | `tisiago.truth` |
| **baseline** | Frozen AlphaGenome embeddings + a logistic head, one label per codon. The comparator. | `tisiago.baseline` |
| **floor** | One-hot codon / ±20 nt Kozak features + the same head. What every model must beat. | `tisiago.baseline` (arms `floor.*`) |

Plan and status: [`docs/PLAN.md`](docs/PLAN.md). Results: [`FINDINGS.md`](FINDINGS.md). What transfers to a
new recipe: [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md). Evaluation rules:
[`docs/INFERENCE_AND_EVALUATION.md`](docs/INFERENCE_AND_EVALUATION.md). What the label pipeline hands over:
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).

## Upstream

```
label pipeline (Ribo-TISH fork)  ->  P-site tracks + reproducible starts  ->  tisiago
swissisoform                     ->  candidate manifest (baseline only)  ->  tisiago
gruyerenome                      ->  AlphaGenome forward passes (baseline extraction)
alphagenome-ft                   ->  AlphaGenome fine-tuning (tracks)
```

## Install and verify

```bash
conda create -n tisiago -c conda-forge python=3.11 uv pip -y
eval "$(conda shell.bash hook)" && conda activate tisiago
uv pip install -e ".[dev,extract,tracks]"
python -m pytest                     # CPU suite; JAX model tests skip here
```

| Stage | Environment |
|---|---|
| Baseline evaluation, truth, metrics (CPU) | conda `tisiago` |
| Baseline extraction (GPU) | conda `alphagenome` (tisiago + gruyerenome editable, pyfaidx) |
| Tracks training and prediction (GPU), JAX tests | uv `.venv/agft`, built by `scripts/run_tracks_env_setup.sh` from `requirements-agft.lock` |

Site paths (genome, GTF, AlphaGenome weights, Ribo-seq data, per-stage Python, partitions) live in a
gitignored `.env`; `.env.example` lists the keys. Every Slurm script sources `scripts/_common.sh`, which
loads `.env` and writes logs to `logs/`. Submit from the repo root.

## Run

**tracks** — reference recipe `tracks.spliced.flux.pos`, per held-out line (`HeLa` shown), 3 seeds:

```bash
sbatch --partition=20 --array=0-35%12 scripts/run_tracks_build_targets.sh   # P-site tracks from BAMs
sbatch scripts/run_tracks_train_genomic.sh --arm rnacond:HeLa \
    $(cat configs/tracks/genomic_fc2d.args) --out data/tracks/runs/tracks.genomic.free.pos.ho-HeLa.s0
sbatch scripts/run_tracks_txcache.sh build --name genomic_fc2d_hela \
    --checkpoint data/tracks/runs/tracks.genomic.free.pos.ho-HeLa.s0/best
python -m tisiago.tracks.cache_seq --genome "$TISIAGO_GENOME" data/tracks/txcache/genomic_fc2d_hela
sbatch scripts/run_tracks_train_spliced.sh --cache data/tracks/txcache/genomic_fc2d_hela \
    --init data/tracks/runs/tracks.genomic.free.pos.ho-HeLa.s0/best --heldout HeLa \
    --out data/tracks/runs/tracks.spliced.flux.pos.ho-HeLa.s0   # head: configs/tracks/spliced_flux_pos.yaml
sbatch scripts/run_tracks_scorecard.sh ...                     # M3 guardrails
sbatch scripts/run_calling_track_score.sh --split val          # caller-free start score
```

Script headers give the full options; GPU partitions follow `.env`.

**baseline** — AlphaGenome extraction, store, multi-seed evaluation against the floors:

```bash
bash scripts/run_baseline_pipeline.sh 20
sbatch scripts/run_baseline_evaluate.sh
```

Run IDs follow `tracks.<stage>.<chx>.<cond>.ho-<line>.s<seed>`; see `docs/PLAN.md` §1.
