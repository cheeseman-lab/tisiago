#!/bin/bash
# One autoresearch iteration: train -> predict -> evaluate, on a CPU node via srun.
#
# The autoresearch loop calls this after editing the CONFIG block in train_experiment.py.
# OBJECTIVE (env) selects which val metric evaluate.py echoes as the climbed `objective:`.
# STORE (env) is the absolute path to the shared embedding store (default: main checkout).
#
# Usage:  OBJECTIVE=winrate64 bash run.sh
set -euo pipefail

STORE="${STORE:-/lab/barcheese01/mdiberna/tisiago/data/store}"
OBJECTIVE="${OBJECTIVE:-auprc}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# ~4 GB for the headline feature set; the full 14-key concat (52k dim) needs ~40 GB.
srun --partition=20 --cpus-per-task=4 --mem=64G --time=0:20:00 \
    bash -c "
        eval \"\$(conda shell.bash hook)\" && conda activate tisiago
        cd '$HERE'
        python train_experiment.py --store '$STORE' --out preds.npz
        OBJECTIVE='$OBJECTIVE' python evaluate.py --preds preds.npz
    "
