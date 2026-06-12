# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Project

**tisiago** v0.1.0 — the TIS-prediction orchestrator. The third repo in the
fry-python-tools translation-initiation pipeline:

```
swissisoform  →  manifest.parquet  →  tisiago (tiling)  →  gruyerenome.embed_positions  →  tisiago (store + heads)
   makes the table      candidates           windows            per-candidate vectors          training/eval
```

- **swissisoform** makes the table: `manifest.parquet`, one row per candidate
  translation-initiation codon (called TIS = positives + matched in-transcript
  negatives, with labels, expression flags, region class, chromosome split).
- **gruyerenome** generates embeddings: generic `embed_positions(sequences, positions)`
  over AlphaGenome / Evo2. No TIS knowledge.
- **tisiago** (this repo) is the orchestrator: reads the table, tiles candidates into
  genomic windows, calls gruyerenome to slice per-candidate vectors, assembles a
  row-aligned vector store, and trains/evaluates lightweight downstream heads.

The central design principle: **decouple the forward-pass window from what's
persisted** — embed over a generous genomic window, store only the vector at each
candidate codon. Every candidate (positive and negative) is centred identically, so
position carries no signal; the head discriminates on context.

## Pipeline & commands

```bash
# 1. table comes from swissisoform (separate repo) -> data/manifest.parquet
# 2. extract per-candidate embeddings (GPU) + assemble store
bash scripts/run_tis_pipeline.sh 20        # 3 extraction arrays -> assemble
# 3. evaluate downstream heads (CPU)
python -m tisiago.eval --store data/store --keys alphagenome_jax/L16k/decoder_1bp/off0.npy
```

## Code Structure

```
tisiago/
├── src/tisiago/
│   ├── tiling.py     # grid-snap genomic windowing (nearby candidates share a forward pass)
│   ├── extract.py    # genome fetch + gruyerenome.load_backend().embed_positions + write per-shard parts
│   ├── store.py      # assemble shard parts into row-aligned [N, D] .npy feature arrays
│   └── eval.py       # logistic / MLP heads, stratified (canonical vs alt-TIS) + near-neighbour resolution test
├── configs/          # tis_alphagenome_16k/131k.yaml, tis_evo2_8k.yaml (gruyerenome backend configs)
├── scripts/          # SLURM array wrappers + layer probe
└── data/             # manifest.parquet + store/ (gitignored)
```

## Environments

tisiago runs across stages in different conda envs:

- **Extraction (GPU)** — runs in the `alphagenome` / `evo2` envs. Needs `tisiago` +
  `gruyerenome` (both editable) + `pyfaidx`:
  ```bash
  conda activate alphagenome   # or evo2
  uv pip install -e ".[extract]"
  uv pip install -e /lab/barcheese01/mdiberna/gruyerenome
  ```
  Evo2 needs `export HF_HOME=/lab/barcheese01/mdiberna/gruyerenome/weights/.hf_cache`
  (the SLURM scripts set this; home dir is over quota).
- **Eval / training (CPU)** — the `tisiago` env (no gruyerenome / no GPU needed):
  ```bash
  conda create -n tisiago -c conda-forge python=3.11 uv pip -y
  conda activate tisiago && uv pip install -e ".[dev]"
  ```

## Store layout

Row-aligned to `manifest.row_idx` (row i = candidate with row_idx == i):

```
data/store/
  manifest.parquet
  config.yaml                                   # provenance: dims, coverage
  embeddings/
    alphagenome_jax/L16k/decoder_1bp/off0.npy   # [N, 1536] fp16
    alphagenome_jax/L131k/decoder_1bp/off0.npy
    evo2/W8k/blocks.28.mlp.l3/off6.npy          # [N, 4096] fp16  (3 layers x offsets {0,3,6,9})
    ...
```

A head experiment = load a few `.npy`, `np.concatenate(axis=1)`, filter rows by the
manifest `split` column. No GPU.

## Results so far (held-out chromosomes, logistic head)

Full tables + interpretation + caveats in [`FINDINGS.md`](FINDINGS.md). Summary:

- AlphaGenome 16k 0.844 · Evo2 blk28 0.881 · **AG+Evo2 0.901** AUROC (linear).
- Non-canonical / alternative TIS hold up: 0.886 (uORF 0.95, extension 0.94, dTIS 0.84).
- Near-neighbour (true base-resolution) win-rate @64bp: AG 0.70 · Evo2 0.75 · combined 0.82
  — Evo2 (1 token/bp) carries the nucleotide resolution; AlphaGenome (128bp-upsampled) the
  regional context. **Report the near-neighbour win-rate, not the global AUROC, as the headline.**

## Code Style

- Build: hatchling · Linter: ruff (line-length 100) · Docstrings: Google · Python >=3.11
