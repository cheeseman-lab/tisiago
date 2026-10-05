#!/bin/bash
# One autoresearch iteration: train -> predict -> evaluate, on a CPU node via srun.
#
# The autoresearch loop calls this after editing the CONFIG block in train_experiment.py.
# OBJECTIVE (env) selects which val metric evaluate.py echoes as the climbed `objective:`.
# STORE (env) is the absolute path to the shared embedding store (default: main checkout).
#
# Usage:  OBJECTIVE=winrate64 bash run.sh
set -euo pipefail

OBJECTIVE="${OBJECTIVE:-auprc}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$ROOT/scripts/_common.sh"
STORE="${STORE:-$ROOT/data/store}"
PYTHON="${PYTHON_BIN:-$TISIAGO_CPU_PYTHON}"
require_python "$PYTHON"
SCHEDULER=()
if [[ -n "${TISIAGO_CPU_PARTITION:-}" ]]; then
    SCHEDULER+=(--partition="$TISIAGO_CPU_PARTITION")
fi

# ~4 GB for the headline feature set; the full 14-key concat (52k dim) needs ~40 GB.
srun "${SCHEDULER[@]}" --cpus-per-task=4 --mem=64G --time=0:20:00 \
    bash -c '
        "$1" "$2" --store "$3" --out "$4"
        OBJECTIVE="$5" "$1" "$6" --preds "$4"
    ' _ "$PYTHON" "$HERE/train_experiment.py" "$STORE" "$HERE/preds.npz" \
    "$OBJECTIVE" "$HERE/evaluate.py"
