#!/bin/bash
#SBATCH --job-name=mdiberna_p6utr5_rtish
#SBATCH --partition=20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%x_%j.out
# Ribo-TISH p-values (TIS + frame, no thresholds) at every real K562 val call, on the predicted
# and on the real synthetic-pipeline BAMs, with each side's own TIS background. Usage (repo root):
#   sbatch autoresearch/p6utr5/run_rtish_candidates.sh <calls/K562 dir> <out dir>
set -euo pipefail
SELF="${BASH_SOURCE[0]}"
[[ -n "${SLURM_JOB_ID:-}" ]] && SELF="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
source "$(cd "$(dirname "$SELF")/../../scripts" && pwd)/_common.sh"
R="$1"; OUT="$2"; mkdir -p "$OUT"
awk -F'\t' 'NR>1{print $2"\t"$7"\t"$8}' "$R/real/K562_TIS_predict.txt" | sort -u > "$OUT/candidates.txt"
for side in predicted real; do
    D="$R/$side"
    "$TISIAGO_RIBOTISH" predict -t "$D/tis_rep1.bam,$D/tis_rep2.bam" -b "$D/chx_rep1.bam,$D/chx_rep2.bam" \
        --tispara "$D/tis_rep1.bam.para.py,$D/tis_rep2.bam.para.py" \
        --ribopara "$D/chx_rep1.bam.para.py,$D/chx_rep2.bam.para.py" \
        -g "$R/annotation.chr7.gtf" -f "$TISIAGO_GENOME" -s "$D/tisBackground.txt" \
        -i "$OUT/candidates.txt" --alt --minaalen 3 --tpth 1 --fpth 1 --fspth 1 --fsqth 1 \
        -o "$OUT/$side.txt" -p "${SLURM_CPUS_PER_TASK:-1}"
done
