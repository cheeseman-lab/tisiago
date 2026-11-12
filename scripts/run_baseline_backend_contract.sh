#!/bin/bash
#SBATCH --job-name=mdiberna_tisiago_baseline_backend_contract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Validate gruyerenome's frozen-backend semantics on a real GPU before extraction.
# Usage: sbatch [-p GPU_PARTITION] \
#   scripts/run_baseline_backend_contract.sh \
#   [CONFIG] [LENGTH]

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
CONFIG="${1:-$REPO_ROOT/configs/tis_alphagenome_16k.yaml}"
LENGTH="${2:-4096}"
PYTHON_BIN="${PYTHON_BIN:-$TISIAGO_AG_PYTHON}"
[[ -x "$PYTHON_BIN" ]] || {
    echo "Python not found at $PYTHON_BIN; check TISIAGO_AG_PYTHON in .env" >&2
    exit 2
}
cd "$REPO_ROOT"

ARGS=(--config "$CONFIG" --length "$LENGTH")

echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -c 'import importlib.metadata as m; print(*[f"{p}=={m.version(p)}" for p in ("jax", "gruyerenome", "tisiago")], sep="\n")'
"$PYTHON_BIN" -m tisiago.baseline.backend_contract "${ARGS[@]}"
