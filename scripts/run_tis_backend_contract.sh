#!/bin/bash
#SBATCH --job-name=tisiago_evo2_contract
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
#   scripts/run_tis_backend_contract.sh \
#   [CONFIG] [LENGTH] [PROFILE] [ALLOW_SUFFIX_SENSITIVITY]

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")" && pwd)/_common.sh"
CONFIG="${1:-$REPO_ROOT/configs/tis_evo2_8k_blk28.yaml}"
LENGTH="${2:-4096}"
PROFILE="${3:-0}"
ALLOW_SUFFIX_SENSITIVITY="${4:-0}"
PYTHON_BIN="${PYTHON_BIN:-$TISIAGO_EVO_PYTHON}"
[[ -x "$PYTHON_BIN" ]] || {
    echo "Python not found at $PYTHON_BIN; run run_tis_evo2_env_setup.sh first" >&2
    exit 2
}
cd "$REPO_ROOT"

ARGS=(--config "$CONFIG" --length "$LENGTH")
if [[ "$PROFILE" == "1" ]]; then
    ARGS+=(--profile)
fi
if [[ "$ALLOW_SUFFIX_SENSITIVITY" == "1" ]]; then
    ARGS+=(--allow-suffix-sensitivity)
fi

echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -c 'import importlib.metadata as m, torch; print("torch", torch.__version__); print(*[f"{p}=={m.version(p)}" for p in ("evo2", "vtx", "flash-attn", "gruyerenome", "tisiago")], sep="\n")'
"$PYTHON_BIN" -m tisiago.backend_contract "${ARGS[@]}"
