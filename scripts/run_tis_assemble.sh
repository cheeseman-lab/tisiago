#!/bin/bash
#SBATCH --job-name=tisiago_assemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Gather per-shard extraction partials into the row-aligned vector store.
# Usage: PYTHON_BIN=/path/to/python sbatch scripts/run_tis_assemble.sh \
#   [MANIFEST] [PARTS_DIR] [STORE_DIR] [GLOB]

set -euo pipefail

REPO_ROOT="${TISIAGO_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
MANIFEST="${1:-$REPO_ROOT/data/manifest.parquet}"
PARTS_DIR="${2:-$REPO_ROOT/data/parts}"
STORE_DIR="${3:-$REPO_ROOT/data/store}"
GLOB="${4:-*.npz}"

PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/evo2-next/bin/python}"
[[ -x "$PYTHON" ]] || {
    echo "Python not found at $PYTHON; set PYTHON_BIN to a uv-managed environment" >&2
    exit 2
}

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.store \
    --manifest "$MANIFEST" \
    --parts-dir "$PARTS_DIR" \
    --store-dir "$STORE_DIR" \
    --glob "$GLOB"
