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

## Direction — a general codon→TIS predictor

The end goal is a **general codon → P(initiation) predictor**: given **any codon**
(all 64, not just AUG/near-cognate) in any expressed transcript, emit a calibrated
probability it is a translation-initiation site — including confidently **rejecting
non-starts**, not merely ranking a real start above a few curated decoys. Build it
**from the ground up — broad first, then narrow.**

1. **Where we are (PoC).** A frozen-embedding + linear head *ranks curated candidates*
   (called TIS vs 3:1 matched in-transcript near-cognate decoys) at 0.90 AUROC / 0.82
   near-neighbour win-rate ([`FINDINGS.md`](FINDINGS.md)). Proved the signal exists and
   is linearly decodable. A stepping stone, **not** the deliverable.
2. **The gap to "general".** (a) the *negative* side is untested — ranking metrics never
   demand a confident "no"; (b) it scores curated candidates, not *every* codon; (c) no
   calibration or specificity at the true genome-wide imbalance.
3. **The grounding.** Score **all 64 codons**, but keep **non-cognate codons
   evaluation-only (never trained on)**. Non-cognate initiation is biologically ≈0, so a
   trustworthy predictor must drive them to ≈0 — an abundant, certain-negative,
   out-of-distribution control. "Non-cognate ≈ 0" is the sanity check that the model
   learned initiation biology, not codon identity.
4. **The build (the comprehensive plan).**
   - **Phase 1 — calibration machinery. ✅ DONE** (`src/tisiago/caller.py`). Train on
     `train`, isotonic-calibrate on held-out `val` (chr7), report reliability + Brier +
     **recall at a false-positives-per-transcript budget** on `test`. Pure CPU on the
     curated 3:1 store. Result: AUPRC 0.741; head is *already well-calibrated* (isotonic
     barely moves Brier); recall **0.733 @ ≤1 FP/transcript**. The reusable metric
     functions that Phase 2 plugs into.
   - **Phase 2 — global all-codon calibrated caller.** The deliverable. tisiago grows a
     **scan capability**: read the GENCODE v49 GTF
     (`swissisoform-v2/data/reference/gencode.v49.primary_assembly.annotation.gtf`),
     walk each expressed transcript's exons, enumerate **every codon** (in `mrna_index`
     coordinates — same system the manifest already uses), tile + embed (one forward
     pass per tile slices *all* positions in it — dense scan ≈ same GPU cost as the
     curated run), score with the calibrated head. Evaluate as a caller (recall @ ≤1
     FP/transcript on near-cognate decoys, **at true genome-wide imbalance**) **and** as a
     grounding check (mean P on held-out **non-cognate** codons → expect ≈0).
     swissisoform still owns *positives*; tisiago generates the background.
   - **Phase 3 — autoresearch the head.** Use the `autoresearch` skill to autonomously
     sweep head architectures (logistic → MLP depth/width, regularization, feature-set &
     offset combinations) with proper cross-validation across seeds/splits — make the
     predictor robust, not just a single-seed point estimate.
   - **Phase 4 — TIS efficiency regression (HeLa first).** Move beyond yes/no into
     quantitative initiation: regress the unused per-condition translational-efficiency
     label (`max_norm_HeLa`), restricted to HeLa, on the frozen embeddings.

**Headline metrics going forward:** (1) recall at a fixed false-positives-per-transcript
budget on near-cognate decoys, at true imbalance; (2) non-cognate negative-control score
≈0. Caller-shaped, not the curated-set AUROC.

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

The Phase-2 dense scan produces a parallel `data/scan_store/` (same layout) row-aligned to
`data/scan_manifest.parquet` — *every* codon in held-out transcripts (from
`tisiago.enumerate_codons`, ~7.9M positions test+val), scored by `tisiago.scan_eval` for
recall at true imbalance + the non-cognate≈0 grounding. It is large (~57 GB test-only at
the AG16k+Evo2 headline keys); scan test-only since calibration uses the curated `val`.

## Results so far (PoC — candidate ranking, held-out chromosomes, logistic head)

Full tables + interpretation + caveats in [`FINDINGS.md`](FINDINGS.md). Summary:

- AlphaGenome 16k 0.844 · Evo2 blk28 0.881 · **AG+Evo2 0.901** AUROC (linear).
- Non-canonical / alternative TIS hold up: 0.886 (uORF 0.95, extension 0.94, dTIS 0.84).
- Near-neighbour (true base-resolution) win-rate @64bp: AG 0.70 · Evo2 0.75 · combined 0.82
  — Evo2 (1 token/bp) carries the nucleotide resolution; AlphaGenome (128bp-upsampled) the
  regional context. **Report the near-neighbour win-rate, not the global AUROC, as the headline.**

## Code Style

- Build: hatchling · Linter: ruff (line-length 100) · Docstrings: Google · Python >=3.11
