#!/bin/bash
# Submit fast-loop configs as GPU jobs. Usage (repo root):
#   bash autoresearch/p6fast/submit.sh <partition> <config.yaml> [more.yaml ...]
# One job runs its configs sequentially (they share the ~2 min cache load). Results append to
# autoresearch/p6fast/results.tsv; full records in data/p6/fastloop/runs/. Prints the job id.
set -euo pipefail
cd "$(dirname "$0")/../.."
part="$1"; shift
[[ $# -gt 0 ]] || { echo "no configs" >&2; exit 2; }
mins=$(( 10 + 15 * $# ))  # 30k-step configs take ~13 min each
sbatch --parsable --partition="$part" --job-name=mdiberna_tisiago_p6fast_ar \
    --time="0-$(printf '%02d:%02d:00' $((mins / 60)) $((mins % 60)))" \
    scripts/run_p6_fastloop.sh --results autoresearch/p6fast/results.tsv "$@"
