# tisiago — Embedding Dataset Architecture & Training-Regime Decisions

**Status:** ✅ **Decided** (operon literature review, 2026-06-22) · drafted 2026-06-18
**Purpose:** Two coupled decisions, both judged against the literature: (Part I) how we
shard, assemble, and store GLM embeddings so training/eval runs are *reproducible and fast*;
(Part II) what training-prior and calibration regime gives a trustworthy caller at the true
genome-wide imbalance.

> **Resolution.** Every lean below was reviewed against the literature by operon
> ([`2026-06-18-part3-literature-review.md`](2026-06-18-part3-literature-review.md)) and
> **confirmed**. The per-decision verdicts (with citations + confidence) are recorded inline as
> **Decided** blocks. One negative-control experiment remains open (balanced-train + Saerens
> recalibrate — D1). Summary table at the end of this doc.

---

## 0. Context

Pipeline: `swissisoform` (manifest) → `gruyerenome` (`embed_positions` over AlphaGenome/Evo2)
→ **tisiago** (tile → store → heads). The GLM forward pass is cheap and already works. **The
cost and the irreproducibility live entirely in the layer between "vectors come off the GPU"
and "a dataset the head/autoresearch loop can hammer."**

**The day's lesson, named.** Today's pain was not bad luck; it was one architectural choice:
*materialize row-aligned monoliths.* The dense scan extracts every codon (~62.7M positions),
then `store.py` scatters the shards into `[62.7M, D]` arrays — 179 GB per AlphaGenome key,
**514 GB per Evo2 key**. Every expensive failure today is a symptom: assembly holds arrays in
RAM (OOM, one key at a time on big-mem nodes); reading them back off NFS runs at ~27 MB/s;
fancy-indexed scattered reads are catastrophically slower still. A compounding incident:
`scancel <name>` (positional) silently no-ops, so three dense jobs ran concurrently and
saturated NFS, starving each other — much of the "NFS is pathologically slow" was self-inflicted
contention.

**The result that makes this urgent (FINDINGS §7, 2026-06-18, AG-only).** Training the head *at*
the true imbalance (49:1 dense negatives) instead of the curated 3:1 set lifted recall@≤1FP/tx
from 0.036 → 0.225 (6×) and drove the non-cognate grounding from 0.109 → 0.0004 — on the same
held-out test set at 230:1. The gains are **rank-based** (AUPRC 0.085→0.246), so they are not a
calibration artefact: the negative *distribution* the head trains against matters. This pushes
us toward genuine large-N training, which makes the data layer (Part I) and the training regime
(Part II) one decision.

**What we're asking operon to adjudicate:** the data architecture (A1–A4) against ML
data-loading practice, and the training/calibration science (D1–D3) against imbalance-learning
and genomic-site-prediction literature.

---

## Part I — Data architecture (speed & reproducibility)

Each item: *current state/pain · proposed approach · alternatives · our lean.*

### A1. Sharding contract

- **Current.** `extract` writes one `.npz` per shard, `shard = crc32(transcript_id) % N`, so a
  tile's candidates stay together (one forward pass slices *all* positions in a tile). Members
  are named `backend::ltag::layer::off`, plus a `row_idx` array = each candidate's position in
  the row-aligned scan space (0..N_scan−1).
- **Pain.** `N` differs by backend with no declaration — **AlphaGenome used N=60, Evo2 N=80** in
  the same `scan_parts_allsplits/`. It happens to work (assembly is keyed by `row_idx`, not by
  shard), but it is undocumented and there is no manifest of `{N, scheme, coverage}` to verify a
  scan is complete.
- **Proposed.** Declare `N` and the hash scheme once per scan in `dataset.yaml`; emit a shard
  index (transcripts/row-range per shard + coverage) as provenance; keep `crc32(transcript_id)`
  to preserve tile integrity.
- **Alternatives.** Content-addressed shards; fixed-row-count shards (would split tiles — bad).
- **Lean.** Keep transcript-hash sharding; add the declared contract + shard index.

### A2. Two assembly regimes — the core fix

The monolith path conflates two needs that want opposite designs.

- **Regime A — compact gather (training & curated-eval substrate).** We only ever train/eval on
  *candidate rows* (≤ a few million: train sample + held-out cognate + non-cognate control). So:
  read each shard **once**, scatter only the wanted rows into a compact `[M, D]` store,
  row-aligned to an experiment manifest via its `src_row_idx` provenance. **Never build the
  monolith.** Reference implementation landed today: `scripts/gather_keys_from_shards.py`
  (smoke-tested on a shard; final coverage assert guarantees every wanted row is filled). This is
  how the Evo2 `ag7` substrate is being built — reading the 1.5 TB of shards once instead of
  materializing 2 TB of throwaway monoliths.
- **Regime B — streaming scorer (dense genome-wide eval).** For every-codon evaluation we *do*
  need all N_scan positions — but only once, and only the scalar prediction. Stream shard → apply
  calibrated head → write per-shard preds (tiny) → aggregate. Never assemble; raw shards are
  deletable after scoring.
- **Lean.** Retire the monolith-assembly path (`store.py`'s full-`[N,D]` scatter); gather for A,
  stream for B.
- **Memory lessons (learned the hard way running `ag7`, 2026-06-21).** Two findings that belong
  in any sizing guidance: (1) building a feature matrix as a *parts-list then `concatenate`*
  holds both copies at once (~2× peak) — `_load_full` now **preallocates** the output and fills
  column-blocks (bit-identical, ~half the load peak). (2) **mmap page-cache counts against the
  cgroup memory limit**: real footprint = explicit arrays + working-set of touched store pages,
  so an Evo2 stack (4×33 GB on disk) inflates RSS well beyond the array sizes. Size jobs to
  `arrays + working-set cache`, not arrays alone — this is an argument for the **streaming /
  minibatch** loaders in A3 (drop cache between batches) once training goes large-N.
- **✅ Decided (operon, High).** Adopt Regime B (stream-and-discard) for all genome-wide
  evaluation; **retire the monolith-assembly path.** This is standard genomics practice —
  Enformer/Borzoi/AlphaGenome predict scalar tracks and never materialize hidden states (the
  VEP pattern); whole-genome scalar preds are tiny (62.7M × 4 B ≈ 251 MB). Keep raw shards
  *only* while the head is still being explored; once frozen (Phase 3+), delete shards and
  retain only the scalar preds + the compact training store.

### A3. Storage format & tier — *operon's format call*

- **Current.** `.npy` fp16 per key on shared NFS. Random / fancy-index reads are
  catastrophic; even sequential block reads run ~27 MB/s.
- **Proposed.** Compact store on **fast SSD**:
  **memmap-per-feature-key fp16** + a row-aligned **parquet manifest** (labels, coords, split).
  Each key is a column-group, so an autoresearch run loads only the subset it sweeps. Small set →
  slurp into RAM (1-min logistic); large set → minibatch SGD straight off the memmap. **One
  regime-agnostic format.**
- **Alternatives for operon to weigh.**
  - *Single wide concatenated matrix* — fastest contiguous read, but wastes I/O when sweeps touch
    feature subsets.
  - *Parquet/Arrow columnar* — portable, great metadata, but decode cost and poor random access
    for dense float blocks.
  - *Zarr* — chunked, good for partial/parallel reads, adds a dependency.
  - *WebDataset / MosaicML StreamingDataset* — purpose-built for sharded streaming into minibatch
    SGD; the natural fit **if** D1 commits us to out-of-core dense training.
- **Lean.** memmap-per-key on SSD for Regime A; adopt a streaming format only if D1 selects
  out-of-core dense training as the headline path.
- **✅ Decided (operon, High).** **Memmap-per-key fp16 on SSD.** Embedding floats are
  high-entropy, so Zarr/Parquet compression buys ~nothing and decode is pure overhead; for the
  actual access pattern (select keys → full-key sequential read → concat → train) plain memmap
  is both simplest and fastest. Zarr's niche (concurrent r/w, partial-chunk decode, cloud) does
  not apply to single-process SLURM+SSD training. WebDataset/StreamingDataset stay reserved for
  an out-of-core SGD head — which autoresearch made unlikely (every MLP lost; linear suffices).

### A4. Reproducibility & provenance

- **Current.** Ad-hoc scripts proliferated (`dense_direct`/`500k`/`local`,
  `build_dense_exp_store`); silent `scancel`-by-name left zombie jobs; no single source of truth
  for how an artifact was built.
- **Proposed.** One `dataset.yaml` (scan definition, keys, splits, neg-cap, seed) + one driver
  builds each artifact; a provenance sidecar (dims, coverage, manifest hash, git SHA, backend
  configs) ships with every store; coverage asserts everywhere; `${USER}`-namespaced job names;
  **always `scancel` by job ID.**
- **Lean.** Adopt as the standard; it is cheap and removes today's whole class of failure.

---

## Part II — Training-regime & evaluation science

### Evaluation protocol (near-settled — stated for review)

Headline metrics, regardless of training regime: **recall @ ≤1 FP/transcript at the true
genome-wide imbalance** (on AUG + near-cognate "cognate" codons) and the **non-cognate ≈ 0**
grounding control (never-trained certain-negatives). **Calibration *and* evaluation are always
drawn at the true prior** (B-like), even when training is bounded — the operating-point failure
in FINDINGS §5 was as much a wrong-prior *calibration/metric* as a wrong-prior *training* issue.

### D1. Train prior — the central fork

**Question.** Train *at* the true imbalance (out-of-core, millions of negatives) — or train on a
bounded/balanced set and **recalibrate to the true prior**?

**Evidence so far** — FINDINGS §7, AG-only (AG16k + AG131k + Kozak), same TEST substrate at
230:1, only the training prior changed:

| metric (TEST @ 230:1) | curated-C (train 3:1) | Dense bal (train 49:1) | **Dense none (train 49:1)** |
|---|---|---|---|
| AUPRC (base 0.0043) | 0.0845 | 0.2023 | **0.2456** |
| recall @ ≤1 FP/tx | 0.036 | 0.193 | **0.225** |
| recall @ ≤5 FP/tx | 0.259 | 0.437 | **0.473** |
| recall @ ≤20 FP/tx | 0.533 | 0.682 | **0.688** |
| non-cognate mean p (→0) | 0.109 | 0.0004 | **0.0003** |
| non-cognate p95 | 0.476 | 0.0012 | **0.0008** |
| cognate Brier | 0.039 | 0.0038 | **0.0036** |

Reads: (1) the gains are **rank-based** (AUPRC, recall curve) → no recalibration of the 3:1 head
could produce them; the negative *distribution* matters. (2) The grounding check **flips from
fail to pass** only under imbalance-matched training. (3) At 49:1, **no class-weighting** beats
`balanced`.

> **`ag7` (AG + Evo2 blk28 off{0,3,6,9} + Kozak, 19.6k-dim) — IN (2026-06-21).** Matched `ag`
> vs `ag7` pair at **1M** negatives (the big node was occupied, so 2M was unschedulable; the
> headline AG table above is 2M), best head `Dense(None)`, TEST @ 230:1:
>
> | metric | `ag` (1M) | **`ag7` (1M, +Evo2)** |
> |---|---|---|
> | AUPRC (base 0.0043) | 0.2424 | **0.2959** |
> | recall @ ≤1 FP/tx | 0.247 | **0.309** |
> | recall @ ≤5 FP/tx | 0.419 | **0.500** |
> | non-cognate mean p | 0.0004 | 0.0004 |
>
> **Answer:** Evo2 adds a real ~25% recall / +22% AUPRC lift — but the **grounding was already
> solved by imbalance-matched training alone** (both ≈0.0004). The two levers do different jobs:
> imbalance-matched training teaches non-cognate≈0; Evo2's base resolution adds *cognate*
> discrimination (true start vs near-cognate decoy). Best caller to date: `ag7` Dense(None),
> recall 0.309 @ ≤1FP/tx at true 230:1.

**Unresolved (do not overclaim).** The clean comparison arm — *balanced-train + recalibrate to
the true prior* — has **not** been run; the §7 result is single-seed and AG-only. We know
recalibration alone cannot close the rank-based gap, but we have not measured how much it closes.

**For operon.** Literature on extreme-imbalance training for tabular / frozen-embedding linear
heads: does matched-prior training reliably beat reweighting + post-hoc recalibration, and under
what conditions? Negative-subsampling theory; calibration under prior shift (e.g. Saerens-style
analytic prior correction); whether out-of-core SGD at true imbalance is worth its cost over a
bounded head + recalibration.

> **✅ Decided (operon, High).** **Train at the matched imbalance — recalibration is not a
> substitute.** Prior-correction (Saerens 2002) and undersampling-recalibration (Dal Pozzolo
> 2015) are *monotonic* → they fix calibration but preserve ranking. Since the §7 AUPRC lift
> (0.085→0.246) is rank-based, the problem is a **distribution** shift in the negatives (curated
> near-cognates are a biased subsample of the genome-wide negatives), not a prior/ratio shift —
> so no recalibration of the curated head can recover it. With a frozen representation, the head
> is the only learnable part, so it learns exactly the negatives it sees. **Negative control
> ✅ run (2026-06-22, `scripts/saerens_control.py`):** Saerens-correcting the curated head to the
> true prior recovers Brier (0.044→0.004, ≈ dense) but leaves AUPRC *exactly* unchanged
> (0.1405 vs dense 0.296) — confirming distribution, not prior. Bonus finding: non-cognate
> grounding (mean p 0.114→0.003) is *also* recoverable by recalibration alone, so "non-cognate
> ≈ 0" is a calibration property, not by itself proof of learned biology (see FINDINGS §8).

### D2. Negative sourcing for training

**Question.** Curated **matched near-cognate decoys** (in-transcript, codon-frequency-stratified,
5′UTR+CDS — "plausible starts the ribosome didn't use") vs. **dense genome-wide negatives**
(diverse, abundant) vs. a **hybrid** (matched hard negatives + diverse genome negatives). §7
hints diversity matters — the dense negatives taught a better boundary than the narrow 3:1
matched set.

**For operon.** Hard-negative vs. diverse-negative mining for genomic site prediction; how
TIS-Transformer / TITER-style models construct negatives; whether stratified matched decoys
remain necessary once diverse negatives are present, or invite a codon-identity shortcut.

> **✅ Decided (operon, High).** **Dense genome-wide cognates as the primary training negative;
> curated near-cognates → an *evaluation* stratum only.** The TFBS negative-sampling study
> (Tourne et al. 2026) found genome-sampled negatives beat curated/shuffled, and that
> matched/shuffled negatives *inflate* in-distribution metrics — directly mirroring the curated
> 3:1 set's flattering 0.90 AUROC. TITER / TIS-Transformer use curated negatives universally and
> never report recall at true imbalance (the §5 conflation). A diverse→hard **hybrid curriculum**
> is worth a Phase-3 test but is *not* the priority — §7 already shows Dense(None) > Dense(bal),
> i.e. the model gains more from the raw distribution than from upweighting hard cases.

### D3. Calibration at the true prior

**Question.** Isotonic regression fit on a held-out draw at the true prior vs. **analytic
prior-correction** (closed-form reweighting of balanced-trained probabilities to the deployment
prior).

**For operon.** Which is more reliable at ~230:1 with only ~3.5k positives; sample-size demands
of isotonic at extreme imbalance; reporting calibration (reliability, Brier) honestly at the
true prior.

> **✅ Decided (operon, Medium).** **Train Dense(None), then isotonic-calibrate on the held-out
> dense val draw at the true prior, with a min-bin floor (≥20 positives/bin)** — the 0.43%
> positive rate otherwise yields zero-positive PAV bins. Report Brier + reliability at the true
> prior (as §7 does). Saerens analytic correction is a *sanity check* only (it preserves the
> curated ranking, which is the wrong ranking); Platt scaling is a more-stable 2-param fallback
> if isotonic proves noisy. Medium confidence — the few-positive regime at 230:1 is undertested.

---

## Part III — Explicit literature asks for operon ✅ answered

*All five answered in [`2026-06-18-part3-literature-review.md`](2026-06-18-part3-literature-review.md); verdicts recorded inline above and summarized below.*

1. **Imbalance-matched training vs. reweight+recalibrate** for linear/MLP heads on frozen
   embeddings at 100:1–1000:1 — what does the evidence favor, and when (D1)?
2. **Negative construction** for genomic site / TIS prediction — matched hard negatives vs.
   diverse genome-wide negatives vs. hybrid (D2).
3. **Calibration under prior shift** — isotonic-at-true-prior vs. analytic prior correction at
   extreme imbalance with few positives (D3).
4. **ML dataset/loader formats** for many repeated training runs over millions of rows ×
   ~10⁴ float features with *feature-subset* sweeps — memmap-per-key vs. zarr vs.
   WebDataset/StreamingDataset (A3).
5. Any prior art on **streaming evaluation at genome scale** (score-and-discard) vs. materializing
   per-position embedding stores (A2).

---

## Appendix — current asset inventory (2026-06-18)

| asset | path | size | state |
|---|---|---|---|
| Curated 3:1 store (all 7 keys) | `data/store` | 19 GB | complete (NFS) |
| Dense scan monoliths | `data/scan_store_allsplits` | 361 GB | AG16k + AG131k + Kozak assembled; **Evo2 not** |
| Dense scan shards | `data/scan_parts_allsplits` | ~1.5 TB | AG 60+60 shards, **Evo2 80/80 done** (16–23 GB each), Kozak |
| Compact experiment store | `data/dense_exp_store` | ~160 GB | 4,329,984 rows; **all 7 `ag7` keys in** (AG + Evo2 4×33 GB + Kozak) |
| Dense scan size | — | 62,683,645 positions | TEST eval substrate: 1,823,302 codons (823,302 cognate + 1,000,000 non-cognate; 3,555 positives; 230.6:1) |

**Storage tiers.** shared NFS (~27 MB/s random in the original benchmark) · fast local SSD
(SSD, 23 TB free — proposed home for the compact store) · `/run/user/$UID` (tmpfs, 51 GB).

---

## Decisions summary (operon review, 2026-06-22)

| decision | verdict | basis | confidence |
|---|---|---|---|
| **D1** train prior | Train at matched imbalance; recalibration is not a substitute (rank gap ⇒ distribution, not ratio) | §7 + Saerens 2002, Dal Pozzolo 2015 | High |
| **D2** negatives | Dense genome-wide cognates primary; curated near-cognates → eval stratum | §7 + Tourne et al. 2026 (TFBS) | High |
| **D3** calibration | Isotonic on dense val at true prior, ≥20-pos bin floor; Saerens = sanity check | Saerens 2002 | Medium |
| **A2/B** eval | Stream-and-discard; retire monolith assembly | Enformer/Borzoi/AlphaGenome, VEP | High |
| **A3** format | Memmap-per-key fp16 on SSD; WebDataset only if out-of-core SGD | high-entropy floats; autoresearch (linear suffices) | High |
| **A1/A4** | Declared sharding contract + one `dataset.yaml` driver + provenance/coverage asserts | (engineering; no lit needed) | — |

**Open experiment:** D1 negative control — balanced-train + Saerens-recalibrate (predicted: recovers
Brier, not AUPRC). **Next:** implementation plan for the decided data layer (memmap-per-key on SSD,
gather-for-A / stream-for-B, retire monolith) + the Dense(None) + isotonic-at-true-prior caller.

**Reference implementation already landed:** `scripts/gather_keys_from_shards.py` (Regime A
compact gather), validated and in use for the `ag7` substrate.
