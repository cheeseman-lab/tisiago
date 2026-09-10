#!/bin/bash
#SBATCH --job-name=tisiago_extract
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
#   sbatch --array=0-19%3 --partition="$TISIAGO_AG_PARTITION" \
#       scripts/run_tis_extract.sh TILE_SPEC CONFIG N_SHARDS [MANIFEST] [OUT_DIR]
#
#   TILE_SPEC : ag16k | ag131k | evo2_8k | evo2_8k_s4k | evo2_4k
#   N_SHARDS  : must equal the array size (e.g. 20 for --array=0-19)
# Set PYTHON_BIN to the uv-managed backend environment and TISIAGO_GENOME to
# the indexed reference FASTA.

set -euo pipefail

REPO_ROOT="${TISIAGO_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
TILE_SPEC="$1"
CONFIG="$2"
N_SHARDS="$3"
MANIFEST="${4:-$REPO_ROOT/data/manifest.parquet}"
OUT_DIR="${5:-$REPO_ROOT/data/parts}"

GENOME="${TISIAGO_GENOME:?Set TISIAGO_GENOME to an indexed reference FASTA}"

if [[ "$TILE_SPEC" == ag* ]]; then
    DEFAULT_PYTHON="$REPO_ROOT/.venv/alphagenome/bin/python"
else
    DEFAULT_PYTHON="$REPO_ROOT/.venv/evo2-next/bin/python"
fi
PYTHON="${PYTHON_BIN:-$DEFAULT_PYTHON}"
[[ -x "$PYTHON" ]] || {
    echo "Python not found at $PYTHON; set PYTHON_BIN to a uv-managed backend environment" >&2
    exit 2
}
"$PYTHON" -c "import gruyerenome, pyfaidx" || {
    echo "Install extraction dependencies with uv before submitting this job" >&2
    exit 2
}

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.extract \
    --manifest "$MANIFEST" \
    --tile-spec "$TILE_SPEC" \
    --config "$CONFIG" \
    --genome "$GENOME" \
    --out-dir "$OUT_DIR" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "$N_SHARDS" \
    --resume
