#!/bin/bash
#SBATCH --job-name=tisiago_dense_repr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Matched multi-seed comparison on diverse dense training negatives and the
# held-out cognate test distribution. Requires both W8k and TXP arrays in STORE.
#
# Usage: sbatch scripts/run_tis_dense_representation_eval.sh \
#   [STORE] [OUT_DIR] [SEEDS] [TRAIN_NEGATIVES] [BOOTSTRAP]
# Override PYTHON_BIN to select another uv-managed environment.

set -euo pipefail

REPO_ROOT="${TISIAGO_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
STORE="${1:-$REPO_ROOT/data/dense_exp_store}"
OUT_DIR="${2:-$REPO_ROOT/data/dense_txp_representation}"
SEEDS="${3:-0,1,2,3,4}"
TRAIN_NEGATIVES="${4:-500000}"
BOOTSTRAP="${5:-1000}"
PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/evo2-next/bin/python}"

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
