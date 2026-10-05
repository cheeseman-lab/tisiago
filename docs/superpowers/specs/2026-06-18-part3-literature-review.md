# Literature Review — Part III responses for the Embedding Dataset Architecture Decision Doc

**Reviewer:** operon session 2026-06-18
**Scope:** Five questions from §Part III of the decision doc, grounded in the §7 results.

---

## Overview: the §7 result in context

The headline finding — dense-trained recall 0.225 vs. curated-trained 0.036 at 230:1, with
AUPRC lifting from 0.085 to 0.246 — is a **rank-based** gain. No recalibration of the 3:1
head can produce it, because the class ordering changed. This is consistent with the two
bodies of literature that matter most: (a) negative sampling for genomic site prediction,
and (b) calibration under prior shift. The literatures agree on the mechanism but disagree
on what's fixable post-hoc.

With Evo2 added (ag7 at 1M negatives): recall reaches 0.309 @ ≤1 FP/tx. Evo2 adds cognate
discrimination; imbalance-matched training fixes grounding. These are separable contributions.

---

## D1. Imbalance-matched training vs. reweight + recalibrate

**Question:** For linear/MLP heads on frozen embeddings at 100:1–1000:1, does matched-prior
training reliably beat reweighting + post-hoc recalibration?

**Answer: Yes, under the specific condition that the negative distribution, not just the
ratio, changes between training and deployment.**

The theoretical framework is well established. Saerens, Latinne & Decaestecker
("Adjusting the Outputs of a Classifier to New a Priori Probabilities," *Neural
Computation* 14, 21–41, 2002; DOI: 10.1162/089976602753284446) introduced the
EM-based prior-correction formula that adjusts posterior probabilities when the class prior
shifts between training and deployment. This is a monotonic transformation — it cannot
change the ranking, only the calibration. Dal Pozzolo, Caelen, Johnson & Bontempi
("Calibrating Probability with Undersampling for Unbalanced Classification," IEEE SSCI
2015, 159–166; DOI: 10.1109/SSCI.2015.33) confirmed this analytically and experimentally:
undersampling warps posteriors but does not affect the ranking order.

This is exactly what the §7 result distinguishes. If the only difference between curated and
dense training were the class ratio (3:1 vs. 49:1), the Saerens correction would close the
gap — and AUPRC would be identical between the two training regimes (since AUPRC is
rank-based and prior-correction preserves ranks). **The fact that AUPRC lifts 0.085 → 0.246
proves the issue is not a prior shift. It is a distribution shift in the negative class.**
The curated negatives (144k matched near-cognates) are a different population from the dense
negatives (2M genome-wide cognates). The head trained on curated negatives learned to
separate starts from a narrow, biased sample of non-starts; the head trained on dense
negatives learned to separate starts from the actual population of non-starts.

The two-stage training literature supports this framing. Work on long-tailed recognition
has shown that for frozen representations + linear classifier, class-balanced resampling
at the classifier stage preserves the learned representation's quality. But this assumes
the representation was trained on the full data distribution. When the representation is
truly frozen (as in tisiago — no fine-tuning), the classifier is the only learnable
component, and what it learns depends entirely on what negatives it sees.

**Recommendation for D1:** Matched-prior training is necessary (not just calibration) when
the training negatives are a biased subset of the deployment negatives. This is tisiago's
situation. The clean comparison arm (balanced-train + recalibrate to true prior) should still
be run to quantify how much recalibration alone recovers — predicting it will close the
calibration gap but not the rank gap.

---

## D2. Negative sourcing for training

**Question:** Curated matched hard negatives vs. dense genome-wide negatives vs. hybrid.

**Answer: Genome-sampled negatives outperform curated/shuffled negatives for genomic site
prediction. This is well-established in the TFBS literature and directly parallels tisiago's
§7 finding.**

The key reference is Tourne, De Waele, Vermeirssen & Waegeman, "How Negative Sampling
Shapes the Performance of Transcription Factor Binding Site Prediction Models,"
*Bioinformatics* 42(2), 2026; DOI: 10.1093/bioinformatics/btag048. This paper
systematically evaluated five negative sampling strategies for TFBS prediction: genomic
sampling, shuffling, dinucleotide shuffling, neighborhood sampling, and cell-line-specific
sampling. Their results showed that genomic sampling based on similarity to positives
performed best, while dinucleotide-shuffled negatives — a common practice — performed
poorly and produced inflated performance estimates on training-distribution test sets.

The parallel to tisiago is direct:
- Curated near-cognate decoys ≈ neighborhood sampling (hard, matched, narrow)
- Dense genome-wide cognates ≈ genomic sampling (diverse, real distribution)
- The curated set inflated metrics (AUROC 0.90 on 3:1 test) just as shuffled negatives
  inflated TFBS metrics

The TIS-specific literature uses curated negatives universally:
- TITER (Zhang et al., "Predicting translation initiation sites by deep learning,"
  *Bioinformatics* 33(14), 2017; DOI: 10.1093/bioinformatics/btx247) used same-triplet
  in-transcript negatives
- TIS Transformer and similar methods use context windows with focal loss

Neither reports recall at true genome-wide imbalance. The curated test set design means
their evaluation was on the same biased negative distribution as training — the same
conflation tisiago's §5 exposed.

**The hybrid question** (matched hard negatives + diverse negatives) is open. The TFBS paper
tested "genomic sampling based on similarity to the positives" — which is a form of hybrid
(genome-sampled but filtered for similarity). This outperformed pure random genomic sampling.
For tisiago, the natural hybrid would be: dense genome-wide cognates (the diverse base) with
the curated near-cognates weighted as hard examples. The curriculum learning literature
suggests starting with diverse negatives and progressively adding harder negatives. But §7
already shows that pure diverse negatives (Dense(None)) beat the curated set
comprehensively, and Dense(None) > Dense(balanced) — suggesting the model benefits from
seeing the raw distribution more than from upweighting hard cases.

**Recommendation for D2:** Dense genome-wide cognates as the primary negative source. The
curated near-cognates can be retained as an evaluation stratum (how well does the head rank
within matched pairs?) but should not dominate training. A hybrid curriculum (diverse →
add hard) is worth testing in Phase 3 but is not the priority.

---

## D3. Calibration at the true prior

**Question:** Isotonic regression at true prior vs. analytic prior-correction at ~230:1 with
~3.5k positives.

**Answer: Isotonic at true prior is more reliable, but needs care with bin sizes at extreme
imbalance.**

The Saerens et al. (2002, cited above) analytic correction adjusts posteriors for a known
prior shift. It is fast, deterministic, and requires no held-out data beyond the
class priors. But it assumes the model's posteriors are well-calibrated at the training prior
— a strong assumption that logistic regression approximately satisfies but that fails after
feature-subset sweeps or regularization changes. More critically, the Saerens correction is
a monotonic transform — it preserves the ranking. Since the §7 result shows that the rank
ordering itself changes with training distribution, analytic correction of a curated-trained
head cannot recover the dense-trained head's performance. It can only be applied *after*
dense training, to adjust the dense-trained posteriors from the training prior to the
deployment prior.

For isotonic regression at 230:1: the concern is bin count. With 3,555 test positives spread
across ~823k cognate codons, the positive rate is 0.43%. Isotonic regression with the
default sklearn implementation (PAV algorithm) will produce many bins with zero positives.
The standard fix is to use a held-out dense val set (chr7 in the current split) with
sufficient positives (~2.2k on val) and merge isotonic bins that have too few positives.
Alternatively, Platt scaling (logistic recalibration) requires only 2 parameters and is
more stable at extreme imbalance, though less flexible.

**Recommendation for D3:** Train the head at true imbalance (Dense(None)), then isotonic-
calibrate on a held-out dense val draw at the true prior, with a minimum-bin-size floor
(e.g., ≥20 positives per isotonic bin). Report Brier score and reliability at the true prior
(as §7 already does). The Saerens correction is a useful sanity check (does it move the
curated-trained posteriors in the right direction?) but cannot be the primary calibration
strategy because it preserves the curated ranking, which is the wrong ranking.

---

## A3. Storage format — memmap-per-key vs. zarr vs. WebDataset/StreamingDataset

**Question:** What format for many repeated training runs over M rows × ~10⁴ float features
with feature-subset sweeps?

**Answer: Memmap-per-key on SSD is the right choice for the current regime. Zarr or
WebDataset only become necessary if training goes out-of-core with minibatch SGD.**

The literature on storage formats for dense float embedding stores converges on a clear
hierarchy for the access patterns tisiago needs:

**For slurp-into-RAM training (current, logistic regression on ≤4M rows):**
Per-key memmap files are optimal. The access pattern is: (1) select feature subset (keys),
(2) load entire key into RAM, (3) concatenate, (4) train. This is a sequential full-file
read per selected key — the simplest possible I/O pattern. Memmap here means the OS page
cache handles the read; no decode/decompress overhead. A 2023 empirical study of columnar
storage for ML embedding workloads found that Zarr "incurs a smaller scanning overhead
compared to Parquet and ORC" but noted that "none of the four formats achieves good
compression with vector embeddings" — since embedding floats are high-entropy, compression
buys nothing and decode costs are pure overhead.

For tisiago's autoresearch loop (20+ experiments per objective, each selecting a feature
subset and loading ~4M × D_subset): memmap-per-key means each run reads only the selected
keys. A 7-key sweep (19.6k dim, 4M rows, fp16) is ~157 GB — well within a 192G SLURM
allocation for a full RAM load on SSD. The autoresearch loop already demonstrated ~20
experiments in ~2 hours at this scale; the I/O was not the bottleneck (the NFS was, and
SSD fixes that).

**For out-of-core minibatch SGD (future, if D1 scales beyond in-core):**
WebDataset or MosaicML StreamingDataset become relevant. These formats shard the data into
tar archives with samples pre-packed for sequential streaming, minimizing random I/O. But
tisiago's head is logistic regression (sklearn), not a neural network trained with
mini-batches. Even at 4M rows, logistic regression's L-BFGS solver operates on the full
matrix. The out-of-core path would only matter if the head moves to SGD-trained models
(MLP with PyTorch), which the autoresearch already ruled out (every MLP lost or crashed).

**Zarr's niche:** Zarr adds value when you need (a) concurrent read/write from multiple
processes, (b) partial decompression of chunks, or (c) cloud-native storage (S3/GCS). None
of these apply to tisiago's current workflow (single-process training on a SLURM node with
local SSD). Zarr adds a dependency and decode overhead with no benefit for sequential
full-key reads.

**Recommendation for A3:** Memmap-per-key fp16 on SSD (`/lab/ops_analysis_ssd`). One `.npy`
per feature key, row-aligned to the manifest parquet. Feature-subset sweeps load only the
selected keys. This is the simplest format that's also the fastest for the access pattern.
Revisit if training moves to out-of-core SGD (adopt WebDataset at that point), but the
autoresearch result (linear suffices) makes this unlikely in the near term.

---

## A2/Regime B. Streaming evaluation at genome scale

**Question:** Score-and-discard vs. materializing per-position embedding stores.

**Answer: Stream-and-discard is the standard approach in genomics for genome-wide scoring.
Materializing the full embedding store is an anti-pattern at scale.**

The genomic site prediction field converged on this years ago. Enformer, Borzoi, and
AlphaGenome all predict genome-wide tracks by tiling the genome into overlapping windows,
running the forward pass, and writing the predictions (1D scalar tracks, not the full
embedding). The intermediate hidden states are never materialized to disk — they exist
transiently in GPU memory during the forward pass and are discarded after the output is
computed.

The decision doc's Regime B (stream shard → apply head → write scalar predictions → discard
shard) recapitulates this pattern for the frozen-embedding case: instead of running the
forward pass at score time, you read the pre-computed shard, apply the lightweight head,
write the per-position prediction, and discard the shard. The scalar predictions are tiny
(62.7M × 4 bytes = 251 MB for the whole genome). The shards are the ephemeral intermediate
representation.

This is functionally identical to how Variant Effect Predictor (VEP) pipelines work: score
all variants in a VCF against a model, write the per-variant scores, discard the model
outputs. Nobody materializes Enformer's [1536-dim] hidden state for every position in the
genome and stores it permanently.

The one difference in tisiago's case: the shards are reusable across multiple heads (that's
the point of the store architecture — decouple extraction from scoring). Keeping the raw
shards on disk is justified as long as the head architecture is still being explored. Once
the head is frozen (Phase 3+), the shards can be deleted and only the scalar predictions
(or the compact experiment store for the training rows) need to persist.

**Recommendation for A2:** Adopt Regime B for all genome-wide evaluation. Keep raw shards
only during active head exploration; once the head is frozen, delete shards and retain only
the scalar predictions + the compact training store. The monolith assembly path should be
retired.

---

## Summary of recommendations

| Decision | Recommendation | Confidence |
|---|---|---|
| **D1** (train prior) | Matched-prior training, not just recalibration — the rank gap proves distribution, not ratio, is the issue | High (§7 + TFBS literature) |
| **D2** (negatives) | Dense genome-wide cognates as primary training source; curated as eval stratum | High (TFBS literature + §7) |
| **D3** (calibration) | Isotonic on dense val at true prior, with bin-size floor; Saerens as sanity check | Medium (few-positive regime untested at 230:1) |
| **A3** (format) | Memmap-per-key fp16 on SSD; revisit if out-of-core SGD needed | High (access pattern is simple) |
| **A2/B** (eval) | Stream-and-discard; retire monolith assembly | High (standard practice) |

The unresolved arm — balanced-train + Saerens recalibrate — should be run as a negative
control. Predicting it recovers calibration (Brier) but not rank (AUPRC).

