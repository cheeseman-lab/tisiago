# Inference and evaluation protocol

Written for the `baseline` line (frozen AlphaGenome feeding a small head): changing a window,
stride, orientation, checkpoint or feature offset changes the feature distribution and requires
retraining that head. The rules below (data roles, thresholds, strata, bootstrap) apply equally to
the `tracks` model, which is compared against `baseline.ag` and the `floor.*` arms under the same
protocol. The start metrics M1 / M2 and guardrails M3 are defined in
[`specs/2026-11-11-rescope-native-caller.md`](specs/2026-11-11-rescope-native-caller.md) §4.

## Define the prediction target precisely

The current `label_tis` is the union of TIS calls across experiments and cell lines. It estimates
a **sequence-encoded propensity for a candidate to be observed as a TIS in at least one assayed
context**. It is not a prediction of cell-line-specific ribosome occupancy or read counts.

For cell-line-specific Ribo-seq prediction, use one output per cell line and evaluate only genes
with evidence that the transcript was expressed/assayed in that line. A missing measurement on an
unexpressed gene is missing data, not zero translation. A useful two-part model is:

1. `P(TIS active | sequence, transcript expression, cell line)`; then
2. `E[log1p(Ribo signal) | TIS active, sequence, expression, cell line]`.

Report the sequence-only model separately from any model that consumes measured RNA abundance.
The latter may be a better Ribo-seq predictor, but it answers a different question.

## Leakage-free data roles

Keep these roles disjoint and group every split by transcript (and preferably chromosome):

1. **train** — fit the downstream weights;
2. **calibration** — map raw scores to the deployment prevalence;
3. **operating validation** — choose the threshold for the FP/transcript budget;
4. **test** — apply the frozen model, calibration, and threshold once.

The code now splits validation transcripts between roles 2 and 3 for dense-trained heads. It also
removes repeated `(transcript_id, mrna_index)` rows before fitting or evaluating a curated head.
The historical curated manifest has 1,310 such repeated sites, all positive; retaining them
silently up-weighted those positives.

The negatives are *uncalled* codons, not experimentally proven negatives. Treat the problem as
positive-unlabelled when interpreting probabilities. At minimum, exclude genes without adequate
Ribo-seq/RNA-seq coverage and report sensitivity to the detection threshold.

## Correct use of the frozen representation

- **AlphaGenome:** genomic windows suit its regulatory representation. Its decoded 1-bp embedding
  still carries coarse/regional context, so keep the explicit local Kozak floor and near-neighbour
  evaluation.
- **Orientation:** feed minus-strand loci as reverse complements. Report plus/minus-strand metrics.
- **Splicing:** ribosomes act on mature mRNA; the baseline sees genomic windows. The `tracks` line
  handles this with the `spliced` stage (FINDINGS §15).

Before trusting a backend or changing its dependencies, run the executable contract on a GPU:

```bash
sbatch -p nvidia-A6000-20 scripts/run_baseline_backend_contract.sh configs/tis_alphagenome_16k.yaml
```

It requires sparse extraction to match dense rows and multi-sequence calls to match singleton calls.

## Reporting

Every arm uses identical unique sites, dense negatives, seeds and data roles, and reports:

- AUPRC at the true candidate prevalence;
- recall and precision at a validation-selected FP/transcript threshold (M1 for `tracks`: a fixed
  budget of calls per transcript against reproducible starts);
- oracle recall-vs-budget only as a clearly labelled ranking diagnostic;
- near-neighbour win rate in mature-transcript coordinates, plus/minus-strand results, and
  non-cognate grounding;
- strata: AUG / AAG / AGG / other near-cognate, in-CDS vs outside the CDS;
- transcript-bootstrap 95% confidence intervals and at least five negative-subsample seeds
  (baseline) or three training seeds (tracks).

Select on validation, then run test once (`--i-am-final`).

```bash
bash scripts/run_baseline_pipeline.sh 20            # AlphaGenome extraction -> store
python -m tisiago.baseline.onehot --store STORE     # floor features
sbatch scripts/run_baseline_evaluate.sh STORE OUT   # baseline.ag vs floor.kozak20 / floor.codon12
```

`baseline.evaluate` retains all training positives, varies only the negative sample across seeds,
divides validation transcripts between calibration and threshold selection, and writes
transcript-bootstrap intervals, paired differences against `floor.kozak20`, and deployable head
artifacts.

## Primary references

- [AlphaGenome model and representation ablations](https://www.nature.com/articles/s41586-025-10014-0)
- [scikit-learn probability-calibration guidance](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn decision-threshold tuning example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_tuned_decision_threshold.html)
