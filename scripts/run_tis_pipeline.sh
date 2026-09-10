#!/bin/bash
# Orchestrate the full TIS embedding store: three per-candidate extraction arrays
# (AlphaGenome 16k + 131k and Evo-2 8k) running in parallel, then a
# single assemble job once all three complete.
#
# Assumes data/manifest.parquet already exists (produced by swissisoform).
#
# Usage: bash scripts/run_tis_pipeline.sh [N_SHARDS]
# Required: TISIAGO_GENOME and ALPHAGENOME_WEIGHTS_PATH.
# Optional scheduler settings: TISIAGO_AG_PARTITION, TISIAGO_EVO_PARTITION,
# TISIAGO_AG_CONCURRENCY, and TISIAGO_EVO_CONCURRENCY.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

N="${1:-20}"
LAST=$((N - 1))
: "${TISIAGO_GENOME:?Set TISIAGO_GENOME to an indexed reference FASTA}"
: "${ALPHAGENOME_WEIGHTS_PATH:?Set ALPHAGENOME_WEIGHTS_PATH to the AlphaGenome checkpoint directory}"
AG_PYTHON="${TISIAGO_AG_PYTHON:-$REPO_ROOT/.venv/alphagenome/bin/python}"
EVO_PYTHON="${TISIAGO_EVO_PYTHON:-$REPO_ROOT/.venv/evo2-next/bin/python}"
CPU_PYTHON="${TISIAGO_CPU_PYTHON:-$EVO_PYTHON}"
for executable in "$AG_PYTHON" "$EVO_PYTHON" "$CPU_PYTHON"; do
    [[ -x "$executable" ]] || {
        echo "Python not found at $executable; configure the uv environments first" >&2
        exit 2
    }
done

AG_SCHEDULER=()
EVO_SCHEDULER=()
if [[ -n "${TISIAGO_AG_PARTITION:-}" ]]; then
    AG_SCHEDULER+=(--partition="$TISIAGO_AG_PARTITION")
fi
if [[ -n "${TISIAGO_EVO_PARTITION:-}" ]]; then
    EVO_SCHEDULER+=(--partition="$TISIAGO_EVO_PARTITION")
fi
AG_CONCURRENCY="${TISIAGO_AG_CONCURRENCY:-3}"
EVO_CONCURRENCY="${TISIAGO_EVO_CONCURRENCY:-2}"

A=$(sbatch --parsable --array="0-${LAST}%${AG_CONCURRENCY}" "${AG_SCHEDULER[@]}" \
    --export=ALL,TISIAGO_REPO="$REPO_ROOT",PYTHON_BIN="$AG_PYTHON" \
    scripts/run_tis_extract.sh ag16k configs/tis_alphagenome_16k.yaml "$N")
echo "ag16k   array job: $A"

B=$(sbatch --parsable --array="0-${LAST}%${AG_CONCURRENCY}" "${AG_SCHEDULER[@]}" \
    --export=ALL,TISIAGO_REPO="$REPO_ROOT",PYTHON_BIN="$AG_PYTHON" \
    scripts/run_tis_extract.sh ag131k configs/tis_alphagenome_131k.yaml "$N")
echo "ag131k  array job: $B"

C=$(sbatch --parsable --array="0-${LAST}%${EVO_CONCURRENCY}" "${EVO_SCHEDULER[@]}" \
    --export=ALL,TISIAGO_REPO="$REPO_ROOT",PYTHON_BIN="$EVO_PYTHON" \
    scripts/run_tis_extract.sh evo2_8k configs/tis_evo2_8k.yaml "$N")
echo "evo2_8k array job: $C"

D=$(sbatch --parsable --dependency="afterok:${A}:${B}:${C}" \
    --export=ALL,TISIAGO_REPO="$REPO_ROOT",PYTHON_BIN="$CPU_PYTHON" \
    scripts/run_tis_assemble.sh)
echo "assemble job (after all extraction): $D"
echo "Monitor: squeue -u \$USER"
