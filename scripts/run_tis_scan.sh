#!/bin/bash
#SBATCH --job-name=mdiberna_tis_scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# Phase 2 dense scan: embed the all-codon scan manifest (every codon in held-out
# transcripts, from tisiago.enumerate_codons) through the SAME extract.py path as
# the curated run. Reuses the window/position decoupling: one forward pass per tile
# slices every enumerated position, so GPU cost ~ the curated run.
#
# Usage:
#   sbatch --array=0-19%3 --partition=nvidia-A6000-20 \
#       scripts/run_tis_scan.sh TILE_SPEC CONFIG ENV N_SHARDS [MANIFEST] [OUT_DIR]
#
#   TILE_SPEC : ag16k | ag131k | evo2_8k   (headline set = ag16k + evo2_8k)
#   ENV       : conda env (alphagenome for ag*, evo2 for evo2_8k)
#   N_SHARDS  : must equal the array size
#
# Scope / size: the scan store is large — ~89 GB at 5632 dims (fp16) for test+val,
# ~57 GB for test only. Calibration uses the CURATED val, so the scan only needs
# TEST transcripts; regenerate test-only with
#   python -m tisiago.enumerate_codons --splits test --out data/scan_manifest.parquet
# and/or restrict to the headline specs (ag16k + evo2_8k) to bound the footprint.

set -euo pipefail

TILE_SPEC="$1"
CONFIG="$2"
ENVNAME="$3"
N_SHARDS="$4"
MANIFEST="${5:-./data/scan_manifest.parquet}"
OUT_DIR="${6:-./data/scan_parts}"

GEN=/lab/barcheese01/mdiberna/swissisoform-v2/data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa

# Use the /lab HF cache (evo2_7b lives here; home dir is over quota). AlphaGenome
# loads from a local weights path and ignores this.
export HF_HOME=/lab/barcheese01/mdiberna/gruyerenome/weights/.hf_cache

eval "$(conda shell.bash hook)"
conda activate "$ENVNAME"
python -c "import pyfaidx" 2>/dev/null || uv pip install -q pyfaidx

python -m tisiago.extract \
    --manifest "$MANIFEST" \
    --tile-spec "$TILE_SPEC" \
    --config "$CONFIG" \
    --genome "$GEN" \
    --out-dir "$OUT_DIR" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "$N_SHARDS"
