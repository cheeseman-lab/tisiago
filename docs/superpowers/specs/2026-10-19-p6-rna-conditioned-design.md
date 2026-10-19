# P6 experiment 2 — RNA-conditioned Ribo-seq prediction (design)

Status: approved by Matteo 2026-10-18 ("go on"). Follows experiment 1 (FINDINGS §12, tag `p6-exp1`).

## 1. Question

RNA-seq is cheap to generate for a new cell line; high-quality Ribo-seq is not. Given DNA and a cell
line's RNA track, can a fine-tuned AlphaGenome predict that line's harringtonine (TIS) and
cycloheximide (CHX) Ribo-seq tracks — including for a cell line it never saw in training?

## 2. Model

- alphagenome-ft, `all_folds`, end-to-end fine-tuning exactly as the experiment-1 base models
  (head lr 1e-3, backbone lr 1e-4, warmup 300, cosine, clip 1.0, 30k steps, batch 1, 131 kb windows,
  RC + shift augmentation, Poisson-multinomial loss, expression mask on TIS/CHX).
- **One shared model across cell lines.** Each training example is (window, condition): DNA ->
  backbone; the condition's pooled RNA track (both strands) enters the head as extra input channels.
- Head kind `rnacond`: the tutorial-style MLP (Linear 256 -> relu -> Linear) on
  concat(1-bp embeddings, RNA features). RNA features (computed in JAX from the raw pooled RNA counts,
  so they follow augmentation): per strand log1p(counts) and log1p of box-smoothed counts at widths
  9, 33, 129, 513 nt. 10 channels. Outputs 4 channels: TIS +/-, CHX +/- (no RNA output).
- The backbone sees DNA only; RNA never reaches the backbone.

## 3. Data and protocol

- Conditions: HeLa, K562, U2OS, RPE1_Async, RPE1_Que, RPE1_Sen. Cell-line groups for hold-out:
  HeLa | K562 | U2OS | RPE1 (all three RPE1 conditions together).
- **Leave-one-cell-line-out, 4 folds.** Fold f trains on the other three groups' conditions on the
  training chromosomes (all but chr7/8/9), sampling (window, condition) uniformly over pairs.
- Windows: `data/p6/windows_new`; expression `data/p6/expression/expression.new.parquet`; tracks
  `data/p6/tracks` (K562/U2OS reprocessed).
- Evaluation on val (chr7) for development; test (chr8/9) once at the end with `--i-am-final`.
- For each fold, evaluate on the held-out group (out-of-distribution) and on the training groups
  (in-distribution, same chr7 windows).

## 4. Evaluation (plain units)

Scorecard per condition x library (TIS, CHX), on val windows, expressed genes only (expression mask):
- Pearson correlation between prediction and replicate 2, and between replicate 1 and replicate 2
  (the yardstick), at four scales: gene totals (log1p, MANE CDS), 30-nt bins along MANE transcripts,
  codon (3-nt) bins, 1 nt. Report each model value as % of the replicate value.
- Comparators: (a) **RNA-scaled baseline** — prediction = pooled RNA track x a per-library scalar fit
  on training lines/chromosomes (does the model add anything beyond "ribosomes follow mRNA"?);
  (b) the experiment-1 per-condition base model of the same condition (trained on that line — an
  in-distribution reference); (c) the experiment-1 CHX winner (`resid_chx`) for CHX.
- CHX dominant-frame agreement (existing periodicity metric) reported alongside.
- Example-gene plots (rep1, rep2, model, RNA-scaled) for a handful of genes per held-out line.

## 5. Out of scope here

Start-site calling, Ribo-TISH, periodicity add-ons (they can be layered on later if the RNA-conditioned
base works). Public RNA-seq (non-input) as conditioning is a later robustness test; our RNA tracks are
matched RNA-input from the same lysate — stated as a caveat.

## 6. Tasks

1. `rnacond` head + RNA-input data path + shared multi-condition training with a held-out group
   (train/evaluate/predict plumbing, run.json, tests).
2. Scorecard module (§4) + example-gene plots, tests.
3. Runs: 4 folds (GPU, three lanes), scorecard, report (FINDINGS §13 + a browsable page).
