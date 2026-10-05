# HANDOFF — Option B genome-wide dense training (2026-06-17)

> **Historical record (executed June 2026).** Script signatures and environments have changed
> since; see `CLAUDE.md` for current commands.

Pick up from here. This is the plan; execute it step by step.

## Context

tisiago predicts translation-initiation sites (TIS) from frozen genome-foundation-model
embeddings. The current head was trained on a curated 3:1 balanced set (192k candidates)
and achieves 0.90 AUROC — but collapses to 0.094 recall at true genome-wide imbalance
(230:1). Option B: retrain the head at true imbalance using a genome-wide dense scan of
every codon in every expressed transcript. The autoresearch fleet already found the optimal
feature stack and regularization on the curated set; we keep those and change only the
training distribution.

## Autoresearch winner (use this config for Option B)

- **Feature stack (7-key):** AG16k + AG131k + Evo2 blk28 off{0,3,6,9} + Kozak one-hot ±20bp
- **Head:** LogisticRegression(C=0.00075, max_iter=1000, class_weight="balanced", solver="lbfgs")
- **Note:** autoresearch best-all-rounder used class_weight=None. Option B adds class_weight="balanced"
  because we're training at true imbalance (~200-300:1). The C=0.00075 (heavy L2) is kept.

## What exists on disk

```
data/manifest.parquet                    # curated 192k candidates (TRAINING DATA for Option B too)
data/store/                              # curated store, all 14 keys + onehot/{codon12,kozakW20}.npy
data/scan_manifest.parquet               # test-only dense scan (5M pos, chr8/chr9)
data/scan_manifest_allsplits.parquet     # MIGHT EXIST — genome-wide manifest job was submitted
data/scan_store_ag/                      # AG16k test-only scan store (assembled, working)
data/scan_parts/ag16k_shard*.npz         # 20 shards, test-only, complete
data/scan_parts/evo2_8k_shard00000.npz   # test-only, shard 0 complete (23GB)
data/scan_parts/evo2_8k_shard00001.npz   # test-only, shard 1 complete (18GB)
data/scan_parts/evo2_8k_shard00002.npz   # CORRUPT — delete this (cancelled mid-write, 4GB truncated)
configs/tis_evo2_8k_blk28.yaml          # NEW — blk28-only evo2 config (drops blk24/26, fits 64G)
```

## Step 0: Check if the genome-wide manifest landed

```bash
ls -la data/scan_manifest_allsplits.parquet
# If it exists, verify:
python -c "
import pandas as pd
m = pd.read_parquet('data/scan_manifest_allsplits.parquet')
print(f'Total: {len(m):,} positions, {m.transcript_id.nunique():,} tx')
print(m.groupby('split').size())
"
# Expected: ~65M positions, ~18.5k transcripts, train/val/test splits
```

If it doesn't exist, generate it:
```bash
conda activate tisiago
python -m tisiago.enumerate_codons \
    --manifest data/manifest.parquet \
    --gtf /lab/barcheese01/mdiberna/swissisoform-v2/data/reference/gencode.v49.primary_assembly.annotation.gtf \
    --genome /lab/barcheese01/mdiberna/swissisoform-v2/data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa \
    --splits train val test \
    --out data/scan_manifest_allsplits.parquet
```

## Step 1: Submit genome-wide GPU extraction (AG16k + AG131k + Evo2 blk28)

Three SLURM arrays. Use `data/scan_manifest_allsplits.parquet` as manifest.
Output to `data/scan_parts_allsplits/`.

**AG16k (60 shards, A6000, 64G, trivial):**
```bash
rm -f data/scan_parts_allsplits/ag16k_shard*.npz  # clean start
sbatch --array=0-59%5 --partition=nvidia-A6000-20 --gres=gpu:1 --mem=64G --time=2:00:00 \
    scripts/run_tis_scan.sh ag16k configs/tis_alphagenome_16k.yaml alphagenome 60 \
    ./data/scan_manifest_allsplits.parquet ./data/scan_parts_allsplits
```

**AG131k (60 shards, A6000, 64G, trivial):**
```bash
sbatch --array=0-59%5 --partition=nvidia-A6000-20 --gres=gpu:1 --mem=64G --time=2:00:00 \
    scripts/run_tis_scan.sh ag131k configs/tis_alphagenome_131k.yaml alphagenome 60 \
    ./data/scan_manifest_allsplits.parquet ./data/scan_parts_allsplits
```

**Evo2 blk28 (80 shards, A6000, 64G — fits because blk28-only):**
```bash
sbatch --array=0-79%3 --partition=nvidia-A6000-20 --gres=gpu:1 --mem=64G --time=4:00:00 \
    scripts/run_tis_scan.sh evo2_8k configs/tis_evo2_8k_blk28.yaml evo2 80 \
    ./data/scan_manifest_allsplits.parquet ./data/scan_parts_allsplits
```

**IMPORTANT:** The evo2 config is `tis_evo2_8k_blk28.yaml` (NOT tis_evo2_8k.yaml).
The blk28-only config extracts 4 keys instead of 12, cutting RAM from ~59GB to ~20GB/shard.
extract.py will assert `config.model == spec.backend` — the spec name is still `evo2_8k`
(same tiling geometry), only the layers list changes.

**CHECK:** extract.py selects layers from the config's `evo2_layers` list.
Verify that the blk28-only config produces parts with 4 keys (blk28 × off{0,3,6,9})
not 12. If extract.py ignores the config and hardcodes 3 layers, you need to
also restrict MODEL_SPECS['evo2_8k'] or add a new spec — but this should not
be the case; the config drives layer selection via the gruyerenome backend.

**Monitor:**
```bash
squeue -u mdiberna
# Check a completed shard:
python -c "import numpy as np; d=np.load('data/scan_parts_allsplits/evo2_8k_shard00000.npz'); print(list(d.keys())[:5], len(d.keys()))"
# Should show 4 keys (blk28 × off{0,3,6,9}), not 12
```

## Step 2: Generate one-hot Kozak features for the genome-wide manifest

The curated store has `onehot/kozakW20.npy` (192k × 164). We need the same for
the genome-wide scan manifest (~65M positions). This is CPU-only — read ±20bp
from the genome FASTA at each position, one-hot encode.

Check if `autoresearch/make_onehot.py` or similar exists — the autoresearch
session created the curated one-hot arrays. If so, adapt it for the scan manifest.
If not, write a script:

```python
# kozak_onehot_scan.py — generate one-hot Kozak ±20bp for the scan manifest
import numpy as np
import pandas as pd
from pyfaidx import Fasta

m = pd.read_parquet('data/scan_manifest_allsplits.parquet')
fa = Fasta('/lab/barcheese01/mdiberna/swissisoform-v2/data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa')

COMP = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
W = 20  # ±20bp around the codon-A position
BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

X = np.zeros((len(m), (2*W+1)*4), dtype=np.float16)
for i, row in enumerate(m.itertuples()):
    # gstart is 0-based position of the codon's A on the plus strand
    center = row.gstart
    chrom_len = len(fa[row.chrom])
    s = max(0, center - W)
    e = min(chrom_len, center + W + 1)
    seq = str(fa[row.chrom][s:e]).upper()
    # Pad if needed
    seq = 'N' * (center - W - s) + seq + 'N' * (e - (center + W + 1))
    if row.strand == '-':
        seq = seq.translate(COMP)[::-1]
    for j, base in enumerate(seq):
        if base in BASES:
            X[i, j*4 + BASES[base]] = 1.0
    if i % 1_000_000 == 0:
        print(f'  {i:,}/{len(m):,}')

np.save('data/scan_parts_allsplits/kozakW20_allsplits.npy', X)
print(f'Saved: {X.shape}')
```

This takes ~10 min for 65M rows. Run on a CPU node (`srun --partition=20 --mem=32G`).

## Step 3: Assemble the genome-wide scan store

After ALL GPU shards complete:

```bash
conda activate tisiago
# Assemble each spec separately first to verify, then combine
python -m tisiago.store \
    --manifest data/scan_manifest_allsplits.parquet \
    --parts-dir data/scan_parts_allsplits \
    --store-dir data/scan_store_allsplits
```

Then manually copy the Kozak one-hot into the store:
```bash
mkdir -p data/scan_store_allsplits/embeddings/onehot/
cp data/scan_parts_allsplits/kozakW20_allsplits.npy \
   data/scan_store_allsplits/embeddings/onehot/kozakW20.npy
```

**Verify:**
```python
import numpy as np
import pandas as pd
m = pd.read_parquet('data/scan_store_allsplits/manifest.parquet')
ag16k = np.load('data/scan_store_allsplits/embeddings/alphagenome_jax/L16k/decoder_1bp/off0.npy')
evo2 = np.load('data/scan_store_allsplits/embeddings/evo2/W8k/blocks.28.mlp.l3/off0.npy')
kozak = np.load('data/scan_store_allsplits/embeddings/onehot/kozakW20.npy')
print(f'manifest: {len(m):,}, ag16k: {ag16k.shape}, evo2: {evo2.shape}, kozak: {kozak.shape}')
assert ag16k.shape[0] == len(m)
assert evo2.shape[0] == len(m)
assert kozak.shape[0] == len(m)
```

## Step 4: Write dense_caller.py

Create `src/tisiago/dense_caller.py`. This module:

1. **Trains TWO heads on the curated store** (data/store/, 192k candidates):
   - **Config C** (autoresearch winner): 7-key stack, LogisticRegression(C=0.00075, max_iter=1000, class_weight=None)
   - **Option B**: same 7-key stack, LogisticRegression(C=0.00075, max_iter=1000, class_weight="balanced")
   Both use StandardScaler, isotonic calibration on curated val (chr7).

2. **Evaluates both heads on the genome-wide scan store TEST split** at true imbalance:
   - `recall_at_fp_budget` on cognate test codons (AUG + near_cognate)
   - `grounding_stats` on non-cognate test codons
   - `reliability` for Brier / calibration

3. **Prints a side-by-side comparison table.**

4. **score_demo(demo_store_path)**: loads a small demo store (3 genes), scores every
   codon with both heads, prints per-gene per-codon table.

The 7-key feature stack to load:
```python
KEYS_7 = [
    "alphagenome_jax/L16k/decoder_1bp/off0.npy",
    "alphagenome_jax/L131k/decoder_1bp/off0.npy",
    "evo2/W8k/blocks.28.mlp.l3/off0.npy",
    "evo2/W8k/blocks.28.mlp.l3/off3.npy",
    "evo2/W8k/blocks.28.mlp.l3/off6.npy",
    "evo2/W8k/blocks.28.mlp.l3/off9.npy",
    "onehot/kozakW20.npy",
]
```

Import metric functions from existing modules:
```python
from tisiago.caller import recall_at_fp_budget, reliability, fit_calibrated_head
from tisiago.scan_eval import grounding_stats
```

## Step 5: 3-gene demo (SCN1A, GRIN1, TSC1)

These genes are NOT in any store — they weren't expressed in the training cell lines.
They need fresh extraction.

1. **Enumerate their cognate codons:**
   Use enumerate_codons.py or a targeted script. These genes are:
   - SCN1A (chr2) — Dravet syndrome, voltage-gated sodium channel Nav1.1
   - GRIN1 (chr9, test split) — NMDA receptor, neurodevelopmental disorders
   - TSC1 (chr9, test split) — tuberous sclerosis, tumor suppressor

   They're completely out of sample — zero rows in the curated manifest.
   Enumerate every cognate codon (AUG + near-cognate) across the full transcript
   (5'UTR through 3'UTR). Produce `data/demo_manifest.parquet`.

2. **Extract embeddings (3 tiny jobs):**
   AG16k, AG131k, Evo2 blk28 — one shard each, ~5 min each.

3. **Generate Kozak one-hot** for the demo positions.

4. **Assemble** into `data/demo_store/`.

5. **Score** with `dense_caller.py score_demo` — both heads.

The demo table shows: for each cognate codon in SCN1A/GRIN1/TSC1,
position, codon, region (5UTR/CDS/3UTR), P(initiation) under Config C,
P(initiation) under Option B. Expect: Config C assigns broad, poorly
calibrated probabilities; Option B concentrates probability on plausible
starts and rejects the rest.

## Step 6: Update FINDINGS.md

Add §7 with:
- Option B results (side-by-side vs Config C at true imbalance)
- 3-gene demo table
- Interpretation: training at true imbalance fixes the decision boundary

## Key files to reference

- `ARCHITECTURE.md` — full system architecture
- `CLAUDE.md` — project instructions
- `ROADMAP.md` — phase status (unpause P2, note Option B)
- `FINDINGS.md` — results (add §7)
- `autoresearch/winners.md` — autoresearch results
- `docs/superpowers/specs/2026-06-16-phase2b-dense-training.md` — the Option B spec
- `src/tisiago/caller.py` — metric functions to reuse
- `src/tisiago/scan_eval.py` — grounding_stats function
- `src/tisiago/extract.py` — GPU extraction driver
- `src/tisiago/store.py` — store assembly
- `src/tisiago/enumerate_codons.py` — codon enumeration
- `configs/tis_evo2_8k_blk28.yaml` — NEW blk28-only evo2 config

## Environments

- **Extraction (GPU):** `alphagenome` env for ag16k/ag131k, `evo2` env for evo2_8k
  - Both need tisiago + gruyerenome (editable) + pyfaidx installed
  - Evo2 needs `export HF_HOME=/lab/barcheese01/mdiberna/gruyerenome/weights/.hf_cache`
- **Eval/training (CPU):** `tisiago` env — numpy/pandas/sklearn, no GPU

