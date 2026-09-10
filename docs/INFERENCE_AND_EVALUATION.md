# Frozen-GLM inference and evaluation protocol

This project uses genome foundation models only for frozen inference. The learned task model is
a small downstream classifier or regressor. Changing a window, stride, orientation, model
checkpoint, hidden layer, or feature offset changes the feature distribution and therefore
requires retraining that small head, but never the foundation model.

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

## Correct use of the frozen representations

- **AlphaGenome:** genomic windows are appropriate for its regulatory representation. Its decoded
  1-bp embedding still carries substantial coarse/regional context, so keep the explicit local
  Kozak baseline and near-neighbour evaluation.
- **Evo2:** it is trained autoregressively. Under the intended indexing, `h_i` includes base `i`,
  while `h_(i+3)` is the first selected state that includes the complete codon and its immediately
  downstream base. However, the installed Vortex intermediate states are measurably
  suffix-sensitive, so describe offsets 0/3/6/9 as empirical full-input states rather than exact
  causal summaries. The extractor deduplicates overlapping requested positions and reconstructs
  the same offset features.
- **Orientation:** continue feeding minus-strand loci as reverse complements. Add a required
  plus/minus-strand metric and an RC-consistency test before calling a model production-ready.
- **Splicing:** ribosomes act on mature mRNA, whereas the established Evo2 arm sees genomic
  sequence. `tisiago.extract_transcript` now provides a transcript-oriented, exon-spliced arm. It
  sends the complete mature transcript plus a fixed N tail and gathers all candidate states in one
  input. Complete-transcript input is deliberately independent of which candidates are present in
  a curated or dense manifest. Treat this as an ablation because a genome-trained model may react
  differently to exon junctions. Its keys use `evo2/TXP`, so a `W8k` head cannot be reused
  accidentally.

Before trusting a backend or changing its dependencies, run the executable contract on a GPU:

```bash
python -m tisiago.backend_contract \
  --config configs/tis_evo2_8k_blk28.yaml --length 4096
```

This requires sparse extraction to match dense rows and multi-sequence calls to match singleton
calls. For Evo2 it also audits suffix invariance and current-base sensitivity. The installed
Evo2/Vortex path currently fails exact suffix invariance, so TXP does not use candidate-dependent
prefix truncation or default length padding.

### Reproducible Evo2 environment upgrades

Keep the old shared environment only as a historical baseline. The supported stack lives in an
isolated uv virtualenv. `gruyerenome` remains the dependency and forward-pass authority;
`tisiago` supplies extraction, contracts, and benchmarks:

```bash
sbatch --partition="$TISIAGO_EVO_PARTITION" \
  scripts/run_tis_evo2_env_setup.sh .venv/evo2-next

PYTHON_BIN="$PWD/.venv/evo2-next/bin/python" \
  sbatch --partition="$TISIAGO_EVO_PARTITION" scripts/run_tis_backend_contract.sh \
  configs/tis_evo2_8k_blk28.yaml 4096 1 1
```

The setup job uses `uv pip install` throughout. It source-builds FlashAttention, installs the exact
validated closure in `requirements-evo2.lock`, then adds `gruyerenome` and `tisiago` editable with
`--no-deps` so source updates cannot silently re-resolve the environment. The upstream
FlashAttention wheel needs a newer glibc than the cluster, so the job compiles the pinned release
for A100 (`sm_80`) against the GPU node's CUDA 12.6 toolkit.
It does not modify a shared Python environment. Promote an upgrade only after its semantic
contract and matched runtime benchmark have been compared with the baseline.
The fourth contract argument makes the already-characterized suffix sensitivity non-fatal while
leaving sparse/dense parity, singleton parity, and current-base sensitivity as required gates.
Omit it for any input design that claims exact causal suffix invariance.

After a matched extraction, compare each arm against its baseline artifact. This verifies exact row
alignment and finite features, and computes drift after upcasting the stored float16 arrays:

```bash
python -m tisiago.compare_extractions BASELINE.npz UPGRADED.npz
```

Add `--require-allclose` only when bit-level-equivalent behavior is an explicit promotion
requirement; otherwise carry the measured drift into the trained-head accuracy comparison.

`gruyerenome` now pins Evo2 0.6, Vortex 1.1, Torch 2.7, and FlashAttention 2.8.0.post2 and always
uses the current Evo2 constructor. The prior 0.5 constructor path was removed after the upgraded
contract passed and W8k artifacts were bit-for-bit identical on the matched benchmark shard.

## Inference speed ladder

Safe engineering changes already implemented:

- hash each distinct transcript once during sharding;
- iterate manifest rows without a duplicate namedtuple list;
- order tiles genomically for FASTA locality;
- gather each unique hidden-state position once per tile;
- fill preallocated fp16 shard matrices instead of millions of Python dictionary entries;
- write shards atomically and support validated job resume;
- fold feature standardization into linear coefficients at inference;
- score linear heads one stored feature block at a time;
- compute FP-budget curves in `O(N log N)` rather than repeated full-array masks.

Accuracy-gated frozen-model ablations:

| arm | feature namespace | expected compute effect | risk |
|---|---|---:|---|
| established Evo2 8 kb / 2 kb stride | `evo2/W8k/...` | 1.0× | reference |
| Evo2 8 kb / 4 kb stride | `evo2/W8kS4k/...` | fewer windows (1.36× reduction on one real scan shard) | less consistent placement/context at tile edges |
| Evo2 4 kb / 2 kb stride | `evo2/W4k/...` | about half the work per window | less upstream context |
| official Evo2 Triton kernels | same input arm, numerically distinct features | 1.20× W8k speedup; 7.5× TXP slowdown on one shard | shape/JIT cost and up to 7.91% row-relative L2 drift |
| official Evo2 1B checkpoint | new namespace | potentially large model-speed gain | different layers/dimensions; must reselect representation |
| exon-spliced transcript Evo2 | `evo2/TXP/...` | one full mature transcript per transcript; often far fewer input bases | input-distribution shift at splice junctions |

The first matched A100 measurement used shard 319/1000 (196 candidates, 18 transcripts). W8k
processed 89 windows / 729,088 input bases in 100.9 seconds of model time and 208.5 seconds total;
TXP processed 18 complete transcripts / 67,012 bases in 14.7 seconds of model time and 45.2 seconds
total. That is a measured **6.86x model-time** and **4.61x end-to-end** speedup on this shard. Both
artifacts contain the same row IDs and four finite `(196, 4096)` feature matrices. Treat this as a
systems result only; TXP still requires a separately trained head and accuracy evaluation.

On the same shard, the promoted Evo2 0.6 / Vortex 1.1 non-kernel stack reduced W8k model time to
88.2 seconds and TXP model time to 13.2 seconds. W8k features were bit-for-bit identical to the
old stack. TXP features changed by at most 2.63% row-relative L2, so TXP remains a separately
trained representation rather than a drop-in replacement. Within the upgraded stack, TXP was
**6.68x faster in model time** and **2.91x faster end to end** than W8k.

The opt-in Vortex kernels are shape-sensitive on this workload. They reduced fixed-length W8k
model time from 88.2 to 73.6 seconds (1.20x), but increased variable-length TXP model time from
13.2 to 99.1 seconds (7.5x slower) and changed hidden rows by as much as 7.91% row-relative L2.
Keep `evo2_use_kernels: false` for TXP. The kernel path is only an accuracy-gated W8k ablation,
not the default backend.

Do not mix any ablation's embeddings with a head trained on `W8k`. The distinct namespaces in
`MODEL_SPECS` make that failure noisy.

Once the feature set and linear head are frozen, the production architecture should multiply each
model's selected hidden rows by its corresponding fused coefficient block during extraction and
write only a partial scalar logit. Summing the AlphaGenome, Evo2, and one-hot partial logits before
calibration is mathematically identical to concatenating and storing all 19,620 features, while
reducing persistent inference output to one scalar per candidate. Keep full embeddings only for
experimentation. `LinearHeadArtifact` is the portable, pickle-free contract for this path: it
stores the folded raw-feature coefficients, intercept, isotonic knots, operating threshold, and
the exact ordered feature schema. It can project any model-specific feature subset into an
additive partial logit; the intercept is applied once after the model contributions are summed.

## Experimental decision table

Every frozen-inference arm should use identical unique sites, dense negatives, seeds, and data
roles. Record both accuracy and systems metrics:

- AUPRC at the true candidate prevalence;
- recall and precision at a validation-selected FP/transcript threshold;
- oracle recall-vs-budget only as a clearly labeled ranking diagnostic;
- near-neighbour win rate in mature-transcript coordinates, plus/minus-strand results, and
  non-cognate grounding;
- transcript-bootstrap 95% confidence intervals and at least five negative-subsample seeds;
- model load/JIT time, forward time, windows, input bp, unique positions transferred, output bytes,
  peak host/GPU memory, and end-to-end candidates/second.

Select from the validation Pareto frontier, then run test once. A faster arm should be accepted only
when its confidence interval rules out a scientifically meaningful loss chosen before the run.

Example extraction arms (same frozen Evo2 config, different namespaced tile specs):

```bash
sbatch --array=0-19%3 scripts/run_tis_extract.sh \
  evo2_8k_s4k configs/tis_evo2_8k_blk28.yaml 20

sbatch --array=0-19%3 scripts/run_tis_extract.sh \
  evo2_4k configs/tis_evo2_8k_blk28.yaml 20
```

After building row-aligned stores with the corresponding namespaced keys, evaluate with
`dense_caller --features ag_evo8k_s4k` or `--features ag_evo4k`.

Run the exon-spliced arm on both the curated and dense manifests, then assemble its shards into
the matching stores:

```bash
# Curated train/calibration features.
sbatch --array=0-19%2 --partition="$TISIAGO_EVO_PARTITION" \
  scripts/run_tis_extract_transcript.sh \
  configs/tis_evo2_8k_blk28.yaml 20 data/manifest.parquet data/txp_parts
python -m tisiago.store --manifest data/manifest.parquet \
  --parts-dir data/txp_parts --store-dir data/store \
  --glob 'evo2_txp_shard*.npz'

# One-time dense experiment features (2M diverse train negatives plus held-out val/test).
sbatch --array=0-19%2 --partition="$TISIAGO_EVO_PARTITION" \
  scripts/run_tis_extract_transcript.sh \
  configs/tis_evo2_8k_blk28.yaml 20 \
  data/dense_exp_store/manifest.parquet data/dense_txp_parts
python -m tisiago.store --manifest data/dense_exp_store/manifest.parquet \
  --parts-dir data/dense_txp_parts --store-dir data/dense_exp_store \
  --glob 'evo2_txp_shard*.npz'

# Curated diagnostic, followed by the primary dense-negative comparison.
sbatch scripts/run_tis_representation_eval.sh
sbatch scripts/run_tis_dense_representation_eval.sh

# Required invariant before accepting the dense comparison.
sbatch scripts/run_tis_txp_overlap_contract.sh data/store data/dense_exp_store
```

Both comparison runners retain all training positives, vary only the negative sample across five
matched seeds, divide validation transcripts between calibration and threshold selection, and
write transcript-bootstrap intervals plus deployable head artifacts. The curated comparison is a
representation diagnostic only. The dense runner compares `AG+W8k` and `AG+TXP` on the cognate
candidate distribution used by the caller; non-cognates remain a separately reported grounding
control.

After an `ag_txp` head wins the dense accuracy gate, pass each seed artifact to transcript
extraction with repeated `--head-artifact` arguments (or as trailing arguments to
`run_tis_extract_transcript.sh`). This switches the shard schema to float32
`partial_logit::<artifact-name>` scalars. Complete those logits with the other row-aligned model
blocks and calibrate them without reconstructing the feature matrix:

```bash
python -m tisiago.projected_score \
  --manifest MANIFEST.parquet \
  --parts-dir PROJECTED_TXP_PARTS \
  --embedding-root STORE/embeddings \
  --head-artifact HEAD_seed0.npz --head-artifact HEAD_seed1.npz \
  --out-dir PREDICTIONS
```

`projected_score` adds each artifact's intercept once, applies its stored isotonic calibration,
and writes both per-seed and ensemble probabilities atomically. The ensemble operating threshold
is recorded in the representation run's `run.json` and must be chosen on operating validation,
never reconstructed from test predictions.

The current `gruyerenome` Evo2 backend still loops over sequences inside
`embed_positions`. Consequently, increasing `--batch-size` currently groups equal-length inputs
at the API boundary but does not create a true batched Evo2 forward pass. The transcript arm can
still be substantially cheaper because it removes redundant genomic windows. True model batching
belongs in `gruyerenome`; once implemented there, it must pass `backend_contract` before use here.
Keeping the forward pass in one repository avoids two subtly different Evo2 implementations.

Run a CPU-only biological/workload preflight before requesting GPUs:

```bash
python -m tisiago.extract_transcript \
  --manifest data/manifest.parquet \
  --config configs/tis_evo2_8k_blk28.yaml \
  --genome "$TISIAGO_GENOME" \
  --gtf "$TISIAGO_GTF" \
  --out-dir /tmp/tisiago-txp-preflight --dry-run
```

On the current curated manifest this validates 192,072 candidate codons across 18,588 transcripts.
The full-transcript workload is about 62.9 million bases versus 526.8 million bases for the
established `W8k` tiling (about 8.4× fewer input bases). This is not an 8.4× runtime promise:
Evo2's length scaling, kernel path, and the serialized backend loop must be measured.

The TXP output matrices have the same width as the genomic Evo2 matrices. The 4.33M-row dense
experiment needs about 132 GiB for four fp16 TXP offsets and a high-memory assembly job. That
one-time cost is justified because it lets the head learn from diverse deployment-like negatives.
Do not materialize those vectors for the 62.7M-row production scan: after model selection, project
each feature block to scalar partial logits during extraction and retain only those scores.

## Primary references

- [AlphaGenome model and representation ablations](https://www.nature.com/articles/s41586-025-10014-0)
- [Official Evo2 inference and embedding implementation](https://github.com/ArcInstitute/evo2)
- [scikit-learn probability-calibration guidance](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn decision-threshold tuning example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_tuned_decision_threshold.html)
