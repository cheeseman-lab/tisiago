#!/bin/bash
#SBATCH --job-name=tisiago_assemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Gather per-shard extraction partials into the row-aligned vector store.
# Usage: sbatch scripts/run_tis_assemble.sh \
#   [MANIFEST] [PARTS_DIR] [STORE_DIR] [GLOB]

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
MANIFEST="${1:-$REPO_ROOT/data/manifest.parquet}"
PARTS_DIR="${2:-$REPO_ROOT/data/parts}"
STORE_DIR="${3:-$REPO_ROOT/data/store}"
GLOB="${4:-*.npz}"

PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
[[ -x "$PYTHON" ]] || {
    echo "Python not found at $PYTHON; check TISIAGO_CPU_PYTHON in .env" >&2
    exit 2
}

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.store \
    --manifest "$MANIFEST" \
    --parts-dir "$PARTS_DIR" \
    --store-dir "$STORE_DIR" \
    --glob "$GLOB"
