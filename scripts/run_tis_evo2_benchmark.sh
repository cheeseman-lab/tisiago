#!/bin/bash
#SBATCH --job-name=tisiago_evo2_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Matched end-to-end benchmark of genomic W8k and exon-spliced TXP extraction.
# Run only after run_tis_backend_contract.sh passes for the same environment/config.
# Usage: sbatch [-p GPU_PARTITION] scripts/run_tis_evo2_benchmark.sh \
#   [SHARD_ID] [N_SHARDS] [OUT_DIR] [MANIFEST] [CONFIG]
# Override PYTHON_BIN to compare an isolated environment with the baseline.

set -euo pipefail

REPO_ROOT="${TISIAGO_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
SHARD_ID="${1:-319}"
N_SHARDS="${2:-1000}"
OUT_DIR="${3:-$REPO_ROOT/benchmarks/txp_20260909/extraction}"
MANIFEST="${4:-$REPO_ROOT/data/manifest.parquet}"
CONFIG="${5:-$REPO_ROOT/configs/tis_evo2_8k_blk28.yaml}"

GENOME="${TISIAGO_GENOME:?Set TISIAGO_GENOME to an indexed reference FASTA}"
GTF="${TISIAGO_GTF:?Set TISIAGO_GTF to the matching transcript annotation}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/evo2-next/bin/python}"
[[ -x "$PYTHON_BIN" ]] || {
    echo "Python not found at $PYTHON_BIN; run run_tis_evo2_env_setup.sh first" >&2
    exit 2
}

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR/w8k" "$OUT_DIR/txp"

echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -c 'import importlib.metadata as m, torch; print("torch", torch.__version__); print(*[f"{p}=={m.version(p)}" for p in ("evo2", "vtx", "flash-attn", "gruyerenome", "tisiago")], sep="\n")'

echo "benchmark arm=W8k shard=$SHARD_ID/$N_SHARDS"
"$PYTHON_BIN" -m tisiago.extract \
    --manifest "$MANIFEST" \
    --tile-spec evo2_8k \
    --config "$CONFIG" \
    --genome "$GENOME" \
    --out-dir "$OUT_DIR/w8k" \
    --shard-id "$SHARD_ID" \
    --n-shards "$N_SHARDS" \
    --no-compress

echo "benchmark arm=TXP shard=$SHARD_ID/$N_SHARDS"
"$PYTHON_BIN" -m tisiago.extract_transcript \
    --manifest "$MANIFEST" \
    --config "$CONFIG" \
    --genome "$GENOME" \
    --gtf "$GTF" \
    --out-dir "$OUT_DIR/txp" \
    --shard-id "$SHARD_ID" \
    --n-shards "$N_SHARDS" \
    --no-compress
