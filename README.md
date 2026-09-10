# tisiago

**TIS-prediction orchestrator** — the third repo in the fry-python-tools translation-initiation pipeline.

```
swissisoform  →  manifest.parquet  →  tisiago (tiling)  →  gruyerenome.embed_positions  →  tisiago (store + heads)
```

tisiago reads the candidate table that [swissisoform](https://github.com/cheeseman-lab/swissisoform)
produces, constructs either genomic windows or exon-spliced transcript inputs, calls
[gruyerenome](https://github.com/cheeseman-lab/gruyerenome) to extract frozen per-candidate
embeddings (AlphaGenome + Evo2), assembles a row-aligned vector store, and trains lightweight
downstream heads to predict translation initiation — a track no genome foundation model is
trained to emit.

## Install and verify

```bash
git clone https://github.com/cheeseman-lab/tisiago.git
git clone https://github.com/cheeseman-lab/gruyerenome.git
cd tisiago

uv venv --python 3.11 .venv/dev
uv pip install --python .venv/dev/bin/python -e ".[dev,extract]"
uv pip install --python .venv/dev/bin/python -e ../gruyerenome
.venv/dev/bin/python -m pytest
```

The CPU environment is sufficient for tests, store assembly, and head evaluation. Frozen-model
extraction requires a model-specific GPU environment. The reproducible Evo2 environment is built
with `scripts/run_tis_evo2_env_setup.sh`; AlphaGenome setup is documented in the gruyerenome
README. Gruyerenome owns every foundation-model forward pass; tisiago never carries a second Evo2
implementation.

## Configure a run

Reference data and gated weights are intentionally not embedded in the repository. They must match
the build used to create the manifest:

```bash
cp .env.example .env
# Edit .env, then export its values into this shell.
set -a; source .env; set +a

# swissisoform produces this input table:
test -f data/manifest.parquet

# Submit GPU extraction and dependent assembly. Partition names are optional
# site settings in .env rather than source-code constants.
bash scripts/run_tis_pipeline.sh 20

.venv/dev/bin/python -m tisiago.eval --store data/store \
    --keys alphagenome_jax/L16k/decoder_1bp/off0.npy evo2/W8k/blocks.28.mlp.l3/off0.npy
```

Every Slurm launcher accepts `PYTHON_BIN` and derives its checkout from `TISIAGO_REPO` or
`SLURM_SUBMIT_DIR`; it does not activate a named conda environment or install packages at runtime.
Submit from the repository root, or export `TISIAGO_REPO=/absolute/path/to/tisiago`.

See `CLAUDE.md` for the full pipeline and store layout.
See [`docs/INFERENCE_AND_EVALUATION.md`](docs/INFERENCE_AND_EVALUATION.md) for the frozen-GLM
inference contract, transcript-oriented Evo2 arm, leakage-free calibration/threshold, and
speed-ablation protocol.

The first matched A100 extraction benchmark found the complete-transcript Evo2 arm was 6.86x
faster in model time and 4.61x faster end to end than W8k tiling on the same shard. The promoted
Evo2 0.6 / Vortex 1.1 stack preserved W8k features bit-for-bit and reduced model time further;
within that stack TXP was 6.68x faster in model time. This is a compute result, not yet an accuracy
claim: `TXP` has a distinct feature namespace and requires a new downstream head. The validated
environment is installed with uv from `requirements-evo2.lock`.

The accuracy gate is reproducible through `run_tis_representation_eval.sh` (curated diagnostic)
and `run_tis_dense_representation_eval.sh` (primary dense-negative comparison). These runners keep
all positives, use matched negative-subsample seeds, select thresholds outside test, bootstrap by
transcript, and save portable `LinearHeadArtifact` files for coefficient-fused inference.

## Result

A linear head on frozen AlphaGenome+Evo2 embeddings reaches **0.92 AUROC** on held-out
chromosomes, and discriminates real alternative TIS from adjacent decoy codons at a
**0.83 near-neighbour win-rate** (true single-base resolution) — Evo2 carries the
nucleotide resolution, AlphaGenome the regional context. Both numbers are the autoresearch
best over a 19.6k-dim feature stack (`autoresearch/winners.md`); a 2-key baseline already
reaches 0.90 / 0.82. See [`FINDINGS.md`](FINDINGS.md).

**Dense evaluation:** at the *true genome-wide imbalance* (~230:1), the best frozen-feature
logistic head reached 0.300 recall at the oracle ≤1 FP/transcript point and 0.307 AUPRC.
Tree heads were within noise, pointing to the frozen representation—not classifier capacity—as
the next experimental lever. See [`FINDINGS.md`](FINDINGS.md) and [`ROADMAP.md`](ROADMAP.md).

> **Evaluation note:** historical recall-at-FP numbers optimized the operating threshold on the
> test predictions and are oracle ranking diagnostics. The current caller selects its threshold
> on a disjoint validation subset and applies it unchanged to test. Re-run before using a historical
> value as a deployable performance claim.
