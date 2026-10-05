#!/bin/bash
#SBATCH --job-name=tisiago_repr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Leakage-free, multi-seed comparison of curated W8k and transcript representations.
# This is a representation ablation; dense-distribution training remains required
# before a head is selected for deployment.
#
# Usage: sbatch scripts/run_tis_representation_eval.sh \
#   [STORE] [OUT_DIR] [SEEDS] [TRAIN_NEGATIVES] [BOOTSTRAP]
# Override PYTHON_BIN to select another environment.

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
STORE="${1:-$REPO_ROOT/data/store}"
OUT_DIR="${2:-$REPO_ROOT/data/txp_representation}"
SEEDS="${3:-0,1,2,3,4}"
TRAIN_NEGATIVES="${4:-60000}"
BOOTSTRAP="${5:-1000}"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.representation_eval \
    --store "$STORE" \
    --out-dir "$OUT_DIR" \
    --seeds "$SEEDS" \
    --train-negatives "$TRAIN_NEGATIVES" \
    --bootstrap "$BOOTSTRAP"
