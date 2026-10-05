#!/bin/bash
#SBATCH --job-name=tisiago_dense_repr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Matched multi-seed comparison on diverse dense training negatives and the
# held-out cognate test distribution. Requires both W8k and TXP arrays in STORE.
#
# Usage: sbatch scripts/run_tis_dense_representation_eval.sh \
#   [STORE] [OUT_DIR] [SEEDS] [TRAIN_NEGATIVES] [BOOTSTRAP]
# Override PYTHON_BIN to select another environment.

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
STORE="${1:-$REPO_ROOT/data/dense_exp_store}"
OUT_DIR="${2:-$REPO_ROOT/data/dense_txp_representation}"
SEEDS="${3:-0,1,2,3,4}"
TRAIN_NEGATIVES="${4:-500000}"
BOOTSTRAP="${5:-1000}"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "$REPO_ROOT"
"$PYTHON" -m tisiago.representation_eval \
    --store "$STORE" \
    --out-dir "$OUT_DIR" \
    --arms ag_w8k ag_txp \
    --seeds "$SEEDS" \
    --train-negatives "$TRAIN_NEGATIVES" \
    --bootstrap "$BOOTSTRAP" \
    --evaluation-scope dense \
    --test-codon-classes AUG near_cognate
