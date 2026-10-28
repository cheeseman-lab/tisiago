#!/bin/bash
#SBATCH --job-name=mdiberna_p6nclater_sites
#SBATCH --partition=20
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Per-site tables (K562 + HeLa val) for the later-exon NC diagnosis. Submit from the repo root.
set -euo pipefail
SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")/../../scripts" && pwd)/_common.sh"
cd "$REPO_ROOT"
GTF=data/p6/txtrain/export/sbflux_k562_s0/predicted_val_floatresample_s0/calls/K562/annotation.chr7.gtf
for pair in K562:k562 HeLa:hela; do
    c=${pair%%:*}; l=${pair##*:}
    "$TISIAGO_CPU_PYTHON" autoresearch/p6nclater/sites.py --cache data/p6/txcache/fc2d_$l \
        --condition "$c" --runs sbflux_${l}_s0 sbflux_${l}_s1 sbflux_${l}_s2 --gtf "$GTF"
done
