# tisiago

**TIS-prediction orchestrator** — the third repo in the fry-python-tools translation-initiation pipeline.

```
swissisoform  →  manifest.parquet  →  tisiago (input construction)  →  gruyerenome forward  →  tisiago (store + heads)
```

tisiago reads the candidate table that [swissisoform](https://github.com/cheeseman-lab/swissisoform)
produces, constructs either genomic windows or exon-spliced transcript inputs, calls
[gruyerenome](https://github.com/cheeseman-lab/gruyerenome) to extract per-candidate embeddings
(AlphaGenome + Evo2), assembles a row-aligned vector store, and trains lightweight downstream
heads to predict translation initiation — a track no genome foundation model is trained to emit.

## Install and verify

```bash
conda create -n tisiago -c conda-forge python=3.11 uv pip -y
eval "$(conda shell.bash hook)" && conda activate tisiago
uv pip install -e ".[dev,extract]"
python -m pytest
```

The `tisiago` conda env covers tests, store assembly, and head evaluation. GPU extraction uses
the `alphagenome` conda env (tisiago + gruyerenome editable) and the locked Evo2 uv env built by
`scripts/run_tis_evo2_env_setup.sh` (`.venv/evo2-next`, from `requirements-evo2.lock`).
gruyerenome owns every foundation-model forward pass.

## Configure a run

Site paths (reference FASTA/GTF, AlphaGenome weights, HF cache, SSD store, per-stage Python,
partitions) live in a gitignored `.env`; `.env.example` lists the keys. Every Slurm script
sources `scripts/_common.sh`, which loads `.env` and writes logs to `logs/`. Submit from the
repo root.

```bash
set -a; source .env; set +a          # only needed for interactive python -m calls
bash scripts/run_tis_pipeline.sh 20  # GPU extraction arrays -> dependent assembly
python -m tisiago.eval --store data/store \
    --keys alphagenome_jax/L16k/decoder_1bp/off0.npy evo2/W8k/blocks.28.mlp.l3/off0.npy
```

See `CLAUDE.md` for the pipeline and store layout, and
[`docs/INFERENCE_AND_EVALUATION.md`](docs/INFERENCE_AND_EVALUATION.md) for the leakage-free
evaluation protocol, the exon-spliced Evo2 arm, and extraction speed ablations.

## Result

On the curated 3:1 set, a linear head on AlphaGenome+Evo2 embeddings reaches 0.92 AUROC on
held-out chromosomes (one-hot ±20 bp sequence floor: 0.75) and a 0.83 near-neighbour win-rate.

At the **true genome-wide imbalance** (~230:1), the clean protocol (5 negative-subsample seeds,
threshold chosen on validation, transcript-bootstrap CIs) gives the dense-trained 7-key ensemble
**AUPRC 0.304 [0.288, 0.322]** and **recall 0.306 at 0.99 FP/transcript**. Recall is driven by
AUG starts (0.58); near-cognate starts are largely missed (0.07). See [`FINDINGS.md`](FINDINGS.md)
§10 and [`ROADMAP.md`](ROADMAP.md).
