# Autoresearch program — P6 Ribo-seq track model (K562)

## Goal
Find a fine-tuned AlphaGenome track model whose **local CHX periodicity** is closest to the
replicate reference, without losing fit relative to the mlp base (`data/p6/runs/k562_base`).
K562 only (`--arm condition:K562`); trials ~10k steps.

## Hard rules
- **Pick on val (chr7) only; never test.** Trials evaluate `--split val`; `--split test
  --i-am-final` is for the single final winner, after the search ends.
- New data only: `--windows-dir data/p6/windows_new --expression
  data/p6/expression/expression.new.parquet` (the launcher sets both).
- At most 2 GPU jobs at a time (the launcher enforces it; see its header). Every GPU job of
  the user counts, held ones included; pass `--ignore-jobs 11064676` for the held smoke job and
  do not release it while trial lanes are busy.
- One change per trial against a named parent, so each row answers one question.

## Search space (`train.py` flags)
| Lever | Flags | Values to try |
|---|---|---|
| Head kind | `--head` | `mlp`, `conv`, `seqconv` (one-hot + open-frame features) |
| Frame loss | `--frame-weight`, `--frame-block` | 0, 10, 50, 100; block 5, 10, 20 codons |
| Two-stage init | `--init-from <run>/best` | `data/p6/runs/k562_base/best`; head reused only for the same arm and kind |
| Head warm-up | `--freeze-backbone-steps` | 0, 500, 2000 (backbone lr 0 for N steps) |
| Learning rates | `--learning-rate`, `--backbone-learning-rate`, `--warmup-steps` | head 1e-3 / 3e-4; backbone 1e-4 / 3e-5 |
| Length | `--steps` | 10000 (trials); 30000 only for the winner |

## Score and gate
Each trial appends one row to `autoresearch/p6/results.tsv` (`tisiago.seq2func.trial_score`;
definitions in its docstring):

- **Fit:** `val_profile_kl` and `gene_log_pearson` (channel means over all 1,322 val windows).
  `profile_kl_rel` and `log_pearson_rel` (against `data/p6/eval/ceiling_val.tsv`) are
  informational only: the ceiling scores one replicate against the other, so they are not a
  "fraction of ceiling" and are not used in the gate or the score.
- **Gate:** `gate_fit` = KL ≤ base + 0.02 and log-Pearson ≥ base − 0.005, with base =
  `data/p6/runs/k562_base/trial_score.json`. A trial that fails the gate cannot win.
- **Objective:** `periodicity_score` (100 val windows, CDS `all`) = mean of
  `dom_agree / ref_dom_agree`, `1 − |switch_pred_within − switch_obs_within|` and
  `annot_frame0_pred / annot_frame0_obs`, each clipped to [0, 1]. The reference is
  `data/p6/periodicity/reference_condition_k562/regions_val_max100_replicate.tsv`.
  Also read `p3_pred/p3_obs` and `utr5_dom_agree`; they are not in the score.

## Before the first trial (once)
1. Val references exist: `data/p6/eval/{ceiling,uniform}_val.tsv` (windows_new) and the
   replicate periodicity reference above.
2. Score `k562_base` (no `--base`; this writes the base JSON), chained on its train job:
   `bash scripts/run_p6_trial.sh --queue --ignore-jobs 11064676 --after 11125770 --eval-only
   data/p6/runs/k562_base` (same for `k562_frame_w50` / 11125771 as a reference row).
3. `--init-from` needs a finished source run (train.py refuses otherwise): chain trials on the
   source's train job with `--after <jobid>`.

## Add a trial
```bash
bash scripts/run_p6_trial.sh --dry-run <name> <train args...>   # print the sbatch lines
bash scripts/run_p6_trial.sh [--queue] [--after <jobid>] [--ignore-jobs <jobid>] \
    <name> <train args...>                                     # submit the chain
```
Output goes to `data/p6/runs/ar_<name>/`. Name trials after the change, e.g.
`seqconv_w50_2stage`. Read the row in `results.tsv`, then keep or discard the change.
