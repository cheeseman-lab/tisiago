#!/bin/bash
# Outer autoresearch loop for ONE objective, run inside that objective's git worktree.
#
# Each iteration: a fresh `claude -p` proposes ONE new CONFIG (edits train_experiment.py
# only), then THIS shell runs it, parses the metrics, and keeps/discards deterministically.
# State persists in git + results.tsv, so the agent starts fresh-context every time and the
# loop survives context limits / crashes. The four worktrees differ only in OBJECTIVE.
#
# Usage:  OBJECTIVE=winrate64 STORE=/abs/data/store bash autoresearch/launch_loop.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"   # .../autoresearch
ROOT="$(cd "$HERE/.." && pwd)"          # worktree repo root
cd "$ROOT"

export OBJECTIVE="${OBJECTIVE:?set OBJECTIVE (auprc|auroc|recall1fp|winrate64)}"
export STORE="${STORE:-$ROOT/data/store}"
RESULTS="autoresearch/results.tsv"
LOG="autoresearch/run.log"
AGENTLOG="autoresearch/agent.log"

# Best ${OBJECTIVE}_val over kept/baseline rows so far (the bar a new config must beat).
best() {
  awk -F'\t' -v obj="${OBJECTIVE}_val" '
    NR==1 { for (i=1;i<=NF;i++) col[$i]=i; next }
    { v=$(col[obj]); s=$(col["status"]) }
    (s=="keep"||s=="baseline") && (v+0)>m { m=v+0 }
    END { printf "%.4f", m }
  ' "$RESULTS"
}
g() { grep -oP "^$1:\s*\K[0-9.]+" "$LOG" | head -1; }

echo "=== autoresearch loop OBJECTIVE=$OBJECTIVE STORE=$STORE root=$ROOT ==="
iter=0
while true; do
  iter=$((iter+1))
  bar="$(best)"
  echo "=== [$OBJECTIVE] iter $iter $(date '+%F %T') best=$bar ==="
  rm -f "$HERE/.desc"

  # ---- agent step: propose ONE new CONFIG (edits train_experiment.py only) -------------
  claude -p --dangerously-skip-permissions \
    "You are running ONE autoresearch experiment for OBJECTIVE=$OBJECTIVE. Read autoresearch/program.md, autoresearch/results.tsv (full history — do NOT repeat a config already listed), and the current CONFIG block of autoresearch/train_experiment.py (the current BEST config — build on it). Propose ONE new, untried CONFIG change likely to raise ${OBJECTIVE}_val. Edit ONLY the CONFIG block of autoresearch/train_experiment.py. Then write a single short line describing the change to autoresearch/.desc . Do NOT run commands, do NOT git commit, do NOT touch evaluate.py. Output nothing else." \
    >> "$AGENTLOG" 2>&1 || echo "  (claude -p returned nonzero; proceeding with whatever it edited)"

  desc="$(head -1 "$HERE/.desc" 2>/dev/null | tr '\t' ' ' | cut -c1-120)"
  [ -z "$desc" ] && desc="(agent wrote no .desc)"

  # ---- run step: train + evaluate on a CPU node ----------------------------------------
  bash autoresearch/run.sh > "$LOG" 2>&1
  obj="$(g objective)"

  if [ -z "$obj" ]; then
    echo "  CRASH — no objective line. tail:"; tail -4 "$LOG" | sed 's/^/    /'
    git checkout -- autoresearch/train_experiment.py 2>/dev/null
    printf '%s\t%s\t-\t-\t-\t-\t-\t-\t-\t-\t%s\t%s\n' \
      "$OBJECTIVE-$iter" "$OBJECTIVE" crash "$desc" >> "$RESULTS"
    git add "$RESULTS"; git commit -q -m "[$OBJECTIVE] crash i$iter: $desc" || true
    sleep 3; continue
  fi

  if awk "BEGIN{exit !($obj > $bar)}"; then status=keep; else status=discard; fi
  [ "$status" = discard ] && git checkout -- autoresearch/train_experiment.py

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$OBJECTIVE-$iter" "$OBJECTIVE" \
    "$(g auprc_val)" "$(g auroc_val)" "$(g recall1fp_val)" "$(g winrate64_val)" \
    "$(g auprc_test)" "$(g auroc_test)" "$(g recall1fp_test)" "$(g winrate64_test)" \
    "$status" "$desc" >> "$RESULTS"
  git add "$RESULTS" autoresearch/train_experiment.py
  git commit -q -m "[$OBJECTIVE] $status i$iter obj=$obj (bar=$bar): $desc" || true
  echo "  obj=$obj bar=$bar -> $status"
  sleep 3
done
