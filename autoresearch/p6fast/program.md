# P6 fast loss/structure search — driver program

You are the autonomous driver of a fast experiment loop. Run unattended for up to 4 days (until
2026-10-24 18:00 local) or until the stop rule below. NEVER ask the human anything; NEVER stop to
wait for confirmation. Record everything so another session can resume from the files alone.

## Goal

Find the loss functions and input structure that let a model take a cell line's RNA (plus frozen
AlphaGenome sequence context) and predict base-resolution harringtonine (TIS, initiation) and CHX
(elongation) Ribo-seq tracks, on a cell line it never saw (K562 held out). Prefer ideas that are
biologically grounded (they encode how translation works) over arbitrary tweaks: the result is meant
for a paper, so each winning component should have a one-sentence biological rationale.

Read first: `docs/superpowers/specs/2026-10-20-p6-fast-loss-search-design.md`, FINDINGS.md §12-13,
`.superpowers/sdd/2026-10-19-p6-rnacond/task-3-ledger.md` (what the slow search learned),
`.superpowers/sdd/2026-10-20-p6-fastloop/task-4-report.md`, and the module docstring of
`src/tisiago/seq2func/fastloop.py` (all components, config schema).

## The loop (fast tier)

- Cache: `data/p6/txcache/ar_rnacond_k562_v2` (frozen fine-tuned backbone embeddings, transcript
  coordinates; train = non-chr7/8/9, val = chr7). One config trains + scores in ~4 min on one GPU
  (+ ~2 min cache load per job, shared by the configs in that job).
- Write configs to `autoresearch/p6fast/configs/<round>/<name>.yaml` (YAML, merged into fastloop
  DEFAULTS; unique `name`). Submit with
  `bash autoresearch/p6fast/submit.sh <partition> cfg1.yaml cfg2.yaml ...` (4-8 configs per job).
- Results append to `autoresearch/p6fast/results.tsv`; full records in `data/p6/fastloop/runs/`.
- Wait for jobs with a foreground loop (`while squeue -h -j <ids> | grep -q .; do sleep 300; done`).
  Check failures in `logs/mdiberna_tisiago_p6fast_ar_<jobid>.err`.

## GPU budget (per-partition caps; check https://slurmstatus.wi.mit.edu/limits.html once)

- Allowed partitions: `nvidia-A100-20` (2 GPUs/user), `nvidia-A6000-20` (2), `nvidia-L40S-24` (1).
- Before submitting, count your running+pending GPU jobs per partition (`squeue -u $USER -o "%i %j %T %P"`)
  and stay at or under the cap per partition, counting ALL your jobs (another agent's
  `mdiberna_tisiago_p6ar_lane*` and prediction jobs share A6000/L40S until they finish). Prefer A100.
- Never cancel jobs you did not submit. Never hold or modify `mdiberna_p6_eval_smoke` (11064676).

## Score

- `primary` = mean of K562 TIS codon and nt percent of replicate (K562 held out). Higher is better.
- Guards (vs `ref_rnacond_ps` in results.tsv): K562 TIS and CHX `gene` and `bin30` within 1 point;
  K562 CHX `codon`/`nt` not more than 3 points below. A config that fails a guard is not a winner.
- Also watch: K562 `tis_peak_meta` / `tis_peak_argmax0` (real 0.233 / 0.446), CHX `frame`, and seen
  conditions (overfitting to training lines is acceptable here; the held-out K562 score decides).
- Current best: `t3_foc_lw5_ps` primary 64.9. The slow search found local-loss weight 20 better
  than 5 (full model K562 TIS codon 73.4 / nt 71.0) — start by reproducing that in the fast loop.
- Single runs are noisy: before declaring a winner, rerun it and the runner-up with 2 extra seeds
  (`seed:` in config); report mean +- spread. Differences under ~1.5 points are not real.

## Hypothesis families (rotate; log which family each round tests)

1. Competition among starts: local/transcript multinomial weight and block, factorised output
   (block, temperature, grids), multinomial on CHX too.
2. Scanning: `hazard` output (5'->3' survival), its combination with local losses; crop_mode start.
3. Initiation-elongation coupling: `coupling` loss weight; `chx_contrast`.
4. Noise model: `poisson_nt` / `nb` per-nt likelihoods vs multinomial; coverage weighting.
5. Input structure: `rna`, `rna_sense`, `rna_onesided`, `position`, `embedding_white`; trunk depth/width
   (deep trunks were negative in the slow search — test cheaply, do not dwell).
6. CHX periodicity: `frame` loss weight/block on CHX.

You MAY add new components (inputs, trunks, outputs, losses) to `src/tisiago/seq2func/fastloop.py`
when a hypothesis needs one, with a unit test in `tests/test_seq2func_fastloop.py`; run
`.venv/agft/bin/python -m pytest tests/test_seq2func_fastloop.py --basetemp=.cache/pytest-ar -q`
(export TMPDIR=$PWD/.cache/tmp) before using it; commit with the trailer lines below. You MUST NOT
change scoring: `scorecard.py`, `txcache.py`, and the scoring/metric functions in fastloop.py are
fixed. Never use Ribo-TISH calls, CDS annotation, or hand-coded codon/Kozak features as inputs or
training targets (CDS/calls are for metrics only).

## Promotion to the slow tier (full fine-tune)

When a config beats the slow-tier leader's fast-loop equivalent by >= 3 primary points (3-seed mean)
with guards intact, write `autoresearch/p6fast/promote/<name>.md` (config, scores, rationale, and the
exact mapping to rnacond model/train flags if the components exist there, or what is missing). Do not
launch slow-tier runs yourself; the controller does.

## Logging (mandatory)

- `autoresearch/p6fast/ledger.md`: one line per round: timestamp, family, hypothesis, configs, job
  ids, outcome (primary, guards), decision. Update the "Leaderboard" section (top 10 by primary with
  guards) every round, and a "Findings" section with what each family taught (biology in one line).
- Commit configs + ledger + results.tsv every few rounds (stage only autoresearch/p6fast and your
  fastloop edits; never `git add -A`). Commit trailer:
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
- Never /tmp. Never delete anything outside .cache. Never write in /lab/barcheese01/aTIS_data or
  /lab/cheeseman_sequencing. No push/merge. Do not contact anyone.

## Stop rule

Stop when (a) the deadline passes, or (b) 3 consecutive rounds across at least 3 families give no
gain > 1.5 points over the leader (3-seed means). On stop, write a final summary at the top of the
ledger: leader config, 3-seed scores vs references, the components that mattered with biological
rationale, what failed, and the recommended recipe to scale.

NEVER STOP before the stop rule. NEVER ask for confirmation.
