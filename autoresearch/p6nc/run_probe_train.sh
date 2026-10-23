#!/bin/bash
#SBATCH --job-name=mdiberna_p6nc_probetrain
#SBATCH --partition=20
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Q1 train-chromosome probes scored on the chr7 sets of probe.py. Submit from the repo root.
set -euo pipefail
set -a; source .env; set +a
export TMPDIR=$PWD/.cache/tmp
"$TISIAGO_CPU_PYTHON" autoresearch/p6nc/probe_train.py --data data/p6nc "$@"
