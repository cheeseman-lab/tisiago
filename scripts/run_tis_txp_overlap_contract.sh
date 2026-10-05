#!/bin/bash
#SBATCH --job-name=tisiago_txp_overlap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Require shared curated/dense sites to have identical complete-transcript features.
# Usage: sbatch scripts/run_tis_txp_overlap_contract.sh [REFERENCE] [CANDIDATE]

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
REFERENCE="${1:-$REPO_ROOT/data/store}"
CANDIDATE="${2:-$REPO_ROOT/data/dense_exp_store}"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.store_overlap \
    --reference-store "$REFERENCE" \
    --candidate-store "$CANDIDATE" \
    --require-exact
