#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# Per-candidate embedding extraction (one model spec) as a SLURM array job.
# Each array task processes one transcript-sharded slice of the manifest.
#
# Usage:
#   sbatch --array=0-19%3 --partition=nvidia-A6000-20 \
#       scripts/run_tis_extract.sh TILE_SPEC CONFIG ENV N_SHARDS [MANIFEST] [OUT_DIR]
#
#   TILE_SPEC : ag16k | ag131k | evo2_8k
#   ENV       : conda env (alphagenome for ag*, evo2 for evo2_8k) — must have
#               tisiago + gruyerenome installed editable (see CLAUDE.md).
#   N_SHARDS  : must equal the array size (e.g. 20 for --array=0-19)

set -euo pipefail

TILE_SPEC="$1"
CONFIG="$2"
ENVNAME="$3"
N_SHARDS="$4"
MANIFEST="${5:-./data/manifest.parquet}"
OUT_DIR="${6:-./data/parts}"

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
