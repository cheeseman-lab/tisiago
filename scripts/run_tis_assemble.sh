#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_assemble
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Gather per-shard extraction partials into the row-aligned vector store.
# Usage: sbatch scripts/run_tis_assemble.sh [MANIFEST] [PARTS_DIR] [STORE_DIR]

set -euo pipefail

MANIFEST="${1:-./data/manifest.parquet}"
PARTS_DIR="${2:-./data/parts}"
STORE_DIR="${3:-./data/store}"

eval "$(conda shell.bash hook)"
conda activate tisiago

python -m tisiago.store \
    --manifest "$MANIFEST" \
    --parts-dir "$PARTS_DIR" \
    --store-dir "$STORE_DIR"
