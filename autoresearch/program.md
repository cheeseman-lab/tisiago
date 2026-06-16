# Autoresearch program — TIS head on frozen AG+Evo2 embeddings

## Goal
Maximize the objective metric named by the `OBJECTIVE` env var (one of `auprc`, `auroc`,
`recall1fp`, `winrate64`) on the **validation** split, by editing the `CONFIG` block of
`train_experiment.py`. Four loops run in parallel, each with a different `OBJECTIVE`; this
file is shared, the objective differs per worktree.

## The task
Predict whether a candidate codon is a translation-initiation site (TIS) on a **balanced
1:3 set** (positives = Ribo-seq–called TIS; negatives = uncalled near-cognate/AUG codons,
matched in-transcript and codon-frequency–stratified). Inputs are **frozen embeddings**
already on disk — no GPU, no forward passes. A run trains a lightweight head on `train`,
predicts on `val`+`test`, and `evaluate.py` scores it.

## Hard rules
- **Climb the val metric, never test.** `evaluate.py` prints the `objective:` line from a
  *val* metric. `test` metrics are reported for the record only — never select on them.
- **Edit only the `CONFIG` block** of `train_experiment.py` (FEATURE_KEYS, HEAD,
  HEAD_PARAMS, CLASS_WEIGHT, NEG_SUBSAMPLE). Do **not** edit `evaluate.py` (the fixed metric).
- Keep `SEED=0` and the train cap for comparability across experiments.
- Each run must stay <~2 min CPU.

## Levers (in rough order of expected impact)
1. **Feature subset** — the 14 available keys (see `data/store/config.yaml`):
   `alphagenome_jax/L16k`, `alphagenome_jax/L131k` (decoder_1bp/off0); `evo2/W8k/blocks.{24,26,28}.mlp.l3/off{0,3,6,9}`.
   Input selection is historically the largest lever. Try: AG-only, Evo2-only, AG+Evo2,
   different Evo2 layers/offsets, stacking multiple Evo2 offsets, all-14.
2. **Head** — `logistic` (`C`) vs `mlp` (`hidden_layer_sizes`, `alpha`). The PoC found
   linear suffices for AUROC; nonlinearity may help `winrate64` / `recall1fp`.
3. **Class weighting** — `CLASS_WEIGHT="balanced"` or a dict (logistic only).
4. **Train-negative subsample** — `NEG_SUBSAMPLE` (neg:pos ratio within the 1:3 pool).

## Metric notes
- `winrate64` is the **hard** one: real TIS vs a decoy codon within 64 bp in the same
  transcript — pure base-resolution discrimination (baseline 0.82 for AG+Evo2). Evo2's
  1-token/bp resolution should dominate here; AlphaGenome's regional context may favor `auroc`.
- `recall1fp` is noisier (operating-point) — expect more variance run-to-run.

## Baselines & floors (already logged in results.tsv) — judge configs against these
- **one-hot codon** (`onehot/codon12.npy`): test auroc **0.49** (≈chance). The *control* —
  codon identity carries no signal (negatives are codon-matched), so any real gain is context.
- **one-hot Kozak ±20bp** (`onehot/kozakW20.npy`): test auroc **0.75**, winrate64 **0.73**.
  The *floor*: a trivial raw-sequence head. **Embeddings only earn their place by beating
  this**, not by beating chance.
- **AG16k + Evo2 blk28 off0**, logistic: test auroc **0.90**, auprc **0.76**, winrate64 **0.82**.
  The headline reference. Foundation-model lift over raw sequence is 0.75→0.90 auroc,
  0.73→0.82 winrate.

The one-hot keys are also usable *as features* — concatenate them with embeddings to test
whether explicit local sequence complements the learned representation.

## The experiment loop (how the fleet runs)
The loop is driven by `launch_loop.sh` (one process per objective, in its own git worktree).
Each iteration is a **fresh `claude -p` with no memory of prior iterations** — all state lives
in git + `results.tsv`. Per iteration the shell does the deterministic work; the agent does
exactly one thing:

1. **Agent step (you, when invoked):** read this file, `results.tsv` (everything tried so
   far), and the current `CONFIG` block of `train_experiment.py` (this is the current *best*
   config — build on it). Propose ONE new, **untried** CONFIG change likely to raise
   `${OBJECTIVE}_val`. Edit ONLY the CONFIG block. Write a one-line summary to
   `autoresearch/.desc`. Do **not** run anything, commit, or touch `evaluate.py`.
2. **Shell step (automatic):** runs `run.sh` (train→evaluate on a CPU node), parses the
   metrics, compares `${OBJECTIVE}_val` to the running best (over `keep`/`baseline` rows),
   **keeps** (commit, CONFIG advances) if strictly better else **discards**
   (`git checkout -- train_experiment.py`, reverting to best). Every attempt — keep, discard,
   or crash — is appended to `results.tsv` and committed. Then it loops.

So you never see your own past failures as code, only as `results.tsv` rows: **read the
descriptions to avoid repeating a config**. The loop never stops on its own.
