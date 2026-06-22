# tisiago

**TIS-prediction orchestrator** — the third repo in the fry-python-tools translation-initiation pipeline.

```
swissisoform  →  manifest.parquet  →  tisiago (tiling)  →  gruyerenome.embed_positions  →  tisiago (store + heads)
```

tisiago reads the candidate table that [swissisoform](https://github.com/cheeseman-lab/swissisoform)
produces, tiles each candidate translation-initiation codon into a genomic window,
calls [gruyerenome](https://github.com/cheeseman-lab/gruyerenome) to extract a frozen
per-candidate embedding (AlphaGenome + Evo2), assembles a row-aligned vector store, and
trains lightweight downstream heads to predict translation initiation — a track no
genome foundation model is trained to emit.

## Quick start

```bash
# table comes from swissisoform -> data/manifest.parquet
bash scripts/run_tis_pipeline.sh 20      # extract embeddings (GPU) -> assemble store
python -m tisiago.eval --store data/store \
    --keys alphagenome_jax/L16k/decoder_1bp/off0.npy evo2/W8k/blocks.28.mlp.l3/off0.npy
```

See `CLAUDE.md` for the full pipeline, environment setup, and store layout.

## Result

A linear head on frozen AlphaGenome+Evo2 embeddings reaches **0.92 AUROC** on held-out
chromosomes, and discriminates real alternative TIS from adjacent decoy codons at a
**0.83 near-neighbour win-rate** (true single-base resolution) — Evo2 carries the
nucleotide resolution, AlphaGenome the regional context. Both numbers are the autoresearch
best over a 19.6k-dim feature stack (`autoresearch/winners.md`); a 2-key baseline already
reaches 0.90 / 0.82. See [`FINDINGS.md`](FINDINGS.md).

**In progress (Option B):** evaluating that head at the *true genome-wide imbalance* (~230:1)
by scoring every codon in held-out transcripts — does an imbalance-aware variant call real
starts and reject non-cognate codons (≈0)? See [`ROADMAP.md`](ROADMAP.md) and `HANDOFF_OPTION_B.md`.
