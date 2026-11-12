# P6 re-scope — native track caller and a reproducible-start objective

**Date:** 2026-11-11 · **Status:** proposed · **Author:** session review with Ichihara et al. 2026

Supersedes the "Next" items in [`ROADMAP.md`](../../../ROADMAP.md) P6 and the call-cutoff plan at
the end of [`FINDINGS.md`](../../../FINDINGS.md) §16. It changes **what we climb** and **how calls
are made**; it does not change the track model's architecture, which is deferred until the
objective is fixed.

Reference point for everything below: step-2 flux model, leave-one-cell-line-out, test chr8/9
(FINDINGS §15-16).

---

## 0. Where we are — three classes of experiment

The work so far falls into three classes. They are easy to conflate because FINDINGS numbers them
by experiment rather than by class, so this spec states them once and refers to them by number.

| Class | What | Sections | Status |
|---|---|---|---|
| **1** | **Frozen embeddings -> direct start calls.** AlphaGenome + Evo2 frozen, logistic or tree head, one binary label per codon. | §1-10 | **Closed. This is the baseline.** recall 0.306 @ 0.99 FP/transcript, AUPRC 0.304 (5 seeds, clean protocol); AUG 0.58 vs near-cognate 0.07. |
| **2** | **Fine-tuned AlphaGenome -> tracks, one model per condition, DNA only.** 6 channels (TIS/CHX/RNA x strand). | §11-12 | **Abandoned.** Tracks fit and periodicity was recovered, but starts read off the tracks scored 0.035-0.099 against the class-1 baseline's 0.306, and the per-condition models do not transfer to a new line. |
| **3** | **Fine-tuned AlphaGenome -> tracks, one shared RNA-conditioned model, leave-one-cell-line-out.** Plus the transcript-space head, spliced-input backbone and flux CHX output. | §13-17 | **Live. Build from here.** TIS codon 70-106% of replicate by line; Ribo-TISH Jaccard 0.39 (K562, deployed cutoff). |

**What actually changed between class 2 and class 3** was not the prediction target — both predict
tracks. It was (a) one shared model instead of one per condition, (b) the cell line's RNA track as
input, and (c) leave-one-cell-line-out as the evaluation axis. Gene level went 89% -> 98% of
replicate. There has never been a P6 run that predicted starts directly; that is class 1.

**The gap this spec exists to close.** Class 3 is better than class 2 on *track* metrics, which are
well measured. Whether class 3 beats class 1 **at calling starts** is unknown: the only
apples-to-apples measurement is class 2's (0.035-0.099 vs 0.306), and it has not been rerun since.
Everything after class 2 was judged by Ribo-TISH Jaccard on synthetic BAMs, a different metric on a
different substrate. Note also that in class 2, *real* tracks scored 0.254 on that protocol -- below
the class-1 baseline -- so the deficit is partly in reading starts off tracks at all, not only in
model quality. One caveat in class 3's favour: `track_tis_eval.py` scores a hand-specified statistic
(peak over local background), whereas class 1 is a fitted classifier, so that comparison understates
what a *trained* caller on tracks could do. Phase 3 below supplies exactly that, and Phase 1
measures the number.

---

## 1. Why re-scope before the next hill climb

Three findings, all from our own data, say the current objective does not measure what we want.

**1a. The synthetic-BAM adapter is lossy in principle, not just in practice.**
`seq2func/synthetic_bam.py` turns each track count into a **2-bp read** with the para table
`{2: 1}`, chosen so Ribo-TISH's P-site lands on the right base and "a 2-bp read always passes its
splice-compatibility check, so no transcript model is needed". Real RPFs are 20-38 nt with
per-length offsets. So Ribo-TISH's read-length filters, its periodicity statistics and its
transcript assignment all behave differently on the two inputs — on the synthetic path the
transcript model is bypassed entirely.

The round trip measured the cost with **no model involved** (real tracks -> synthetic BAM ->
Ribo-TISH, against Ribo-TISH on the real BAMs): Jaccard 0.85-0.89 in all six conditions, recovering
95-97% of real calls while **adding 8-12%** (§12). That asymmetry — faithful recall, inflated
false positives — is the signature of a distribution mismatch, and it is the ceiling every
predicted-call number inherits. §16's observation that flux CHX made Ribo-TISH over-call 1.2-2.6x
is the same effect from another angle.

This cannot be fixed by a better converter. A P-site track has already discarded read lengths, so
the inversion does not exist.

**1b. The call cutoff does not transfer between cell lines.** §16: real call counts differ ~2x at
similar predicted counts, and a TIS-p cutoff chosen on one line does not carry to another. Since
the north star is a line with *no* initiation profiling, a cutoff that needs the held-out line's
own Ribo-seq is not deployable. Chasing it as a caller-side problem means predicting library
quality from RNA, which §17 showed is not predictable.

**1c. Percent-of-replicate rewards noisy references.** It is `r(model, rep2) / r(rep1, rep2)`, so a
line whose replicates disagree gets a small denominator and a large score — RPE1 reads 179% on CHX
at nucleotide scale (§15). Climbing it partly optimises for matching poor replicate agreement.

---

## 2. Scope

**In:** the data contract for the regenerated data; the objective and its metrics; a caller that
reads tracks natively; the borrowings from Ichihara et al. that survive scrutiny; corrections to
two stale notes in our own docs.

**Out:** encoder/head architecture changes (resume after the objective is fixed — the §15-16 gains
were measured against metrics this spec demotes, so they need re-measuring first); anything
wet-lab; TISCA2 data, which is unreleased.

---

## 3. Phase 0 — data contract

On the regenerated sequencing data and the ribotish-fork `results/v50` release.

1. **Per-replicate tracks are kept, alongside pooled.** Pooled remains the training target; the
   per-replicate tracks are what define truth in §5. Losing them removes the only noise floor we
   have.
2. **Matched RNA-input library per condition is kept.** The model conditions on it, and §13 flags
   that it is RNA-input from the same lysate.
3. **A reproducible-start truth set**, built from the fork's clean calls. A start is truth only if
   both replicates support it (~20% of near-cognate and ~46% of 5'UTR calls reproduce — Principles
   §1). Each start carries:
   - codon, with **AAG and AGG tagged separately** from other near-cognates (§8);
   - region: 5'UTR / annotated / in-CDS in-frame / in-CDS out-of-frame / other;
   - a pause flag from the `chx_pause_*` columns (§17).
4. **Resolve the P-site convention mismatch between the two repos.** This is a live off-by-one:
   - `tisiago/seq2func/psite.py` mirrors **Ribo-TISH 0.2.7** and keeps the minus-strand P-site at
     `canonical + RIBOTISH_MINUS_SHIFT` (= +1), carried through `data.py` and `predict_tracks.py`;
   - the fork's `pipeline/tracks/make_tracks.py:psite_coord` **corrects** it — "on - the returned
     coordinate is one past the base ... so the base is `coord - 1`" — and the fork is
     **0.2.8+cheeseman.1**.

   So model tracks and release bigWigs differ by 1 nt on the minus strand unless one is shifted.
   Confirm which convention each artifact is in, assert it in code, and check whether P-site
   placement changed between upstream 0.2.7 and 0.2.8. Note §10's unexplained minus-strand deficit
   and §11's "after the Ribo-TISH minus-strand +1 correction" — this is the suspect.
5. **Count the differential starts available.** M2 (§4) needs starts whose share reproducibly
   differs between conditions. Count them per condition pair before relying on the metric; RPE1
   async/quiescent/senescent is the only same-line axis.
6. **Splits frozen:** val chr7, test chr8/9, leave-one-cell-line-out with RPE1's three states as
   one group.

*Done when:* the truth set's counts per class per line are written down, the P-site conventions are
asserted equal (or a documented shift is applied), and the differential-start counts exist.

**Dropped from the earlier draft of this plan:** "patch the Ribo-TISH grouping bug so U2OS/RPE1 get
calls" is not a Phase 0 blocker. The fork's pipeline reruns all aTIS conditions with byte-identical
output; the failure in §15 was confined to the chr7 synthetic-BAM path. See §9.

---

## 4. Objective

**Retire as climbing targets:** per-nucleotide and per-codon percent-of-replicate; Jaccard measured
through synthetic BAMs. Both are kept as *guardrails* (M3), not objectives.

**M1 — start calls in an unseen cell line.** Precision and recall against the Phase 0 truth set at
a **fixed budget of calls per transcript**, on a line the model never saw. Reported separately for
AUG vs near-cognate, and in-CDS vs outside the CDS, so pause-like sites cannot register as wins.
A fixed budget makes the metric independent of how many calls a library happens to yield, which is
what broke cutoff transfer in 1b.

**M2 — differential start usage.** For starts whose share reproducibly differs between two
conditions, the fraction where the model predicts the correct direction. Nothing measures this
today, and it is the metric the downstream variant work rests on.

**M3 — guardrails.** Gene-level and 30-nt percent-of-replicate, plus CHX in-frame share. Must not
regress; must not be tuned on.

Decide on val (chr7) with 3 seeds, differences under ~1.5 points are noise, score test once behind
the existing `--i-am-final` gate.

---

## 5. Native track caller

Retire the synthetic BAM. The model emits tracks; the fork emits real P-site tracks in Ribo-TISH's
own convention (`make_tracks.py`); tracks are therefore the common currency, and one caller can run
on both without fabricating reads.

**Borrow Ribo-TISH's statistics, not its I/O.** On a predicted or real track pair:

- TIS enrichment against a local background;
- the in-frame CHX test, computed on the flux output directly;
- a per-abundance-group background, with **empty groups merged rather than estimated** (see §9);
- codon identity and the ±20 nt context.

**Output:** P(start) with a **jointly learned threshold** — the binary logit is the score logit
minus a learned `tau`, as in IncepTIS. The cutoff becomes a model parameter fit on the **training
lines**, never on the held-out line's Ribo-seq. This is the direct answer to 1b.

**Validation that keeps this honest — the three-way decomposition.** Run:

1. Ribo-TISH on real BAMs — the canonical calls (truth).
2. **The native caller on real tracks** — isolates what the *caller* costs.
3. The native caller on predicted tracks — isolates what the *model* costs.

Arm 2 is the gate: if the native caller on real tracks does not reproduce the canonical calls, the
caller is wrong and arm 3 means nothing. This is Principles §7 ("run the caller identically on
predicted and real tracks") applied to a new caller.

**Keep the caller-free route too.** `seq2func/track_tis_eval.py` already scores predicted tracks on
the frozen baseline's protocol with no caller at all. That is the clean comparison against frozen
`ag7`, and it needs no new code. The two routes answer different questions — caller-free for "did
P6 beat the frozen baseline", native caller for "do our calls match canonical calls" — and both
should be reported. **The caller-free number for class 3 has never been run**
(FINDINGS reports it only for class 2, at 0.035-0.099); it is the first thing to measure.

*Done when:* M1 on a held-out line beats both the frozen `ag7` baseline and Ribo-TISH-on-predicted,
using a threshold fit only on training lines, with arm 2 reproducing canonical calls.

---

## 6. Replicates — truth filter, not denominator

The replicate problem is in the metric, not the data.

- **Training** stays on pooled replicates. Pooling buys depth and reduces noise; training on
  individual replicates trains on their noise.
- **Evaluation** uses replicates to decide **which starts are real** (§3.3), then scores the model
  against that filtered truth set. Reproducibility as a filter, not as a scoring denominator.
- **Sharpness is a per-library property** (§17) and is not predictable from RNA. A held-out line
  gets the training lines' average sharpness, which is arguably the correct library-independent
  answer. Stop rewarding the model for matching a specific library's peak concentration.

---

## 7. What we are not borrowing from Ichihara et al.

Their IncepTIS is the same family as our step 2 — a fine-tuned nucleotide encoder (NTv3 100M), a
light decoder head (Attention U-Net), spliced mRNA up to 32,768 nt, a condition channel, and a
continuous per-position target trained with MSE + BCE. The convergence is reassuring but gives us
nothing to copy:

- **Architecture:** no evidence in the paper that NTv3 beats AlphaGenome here; they never compared.
- **Loss:** simpler than ours, on an easier target (a window-derived cosine score, not raw counts).
- **Evaluation:** random 10-fold by transcript ID within one cell line, no reproducibility floor,
  no sequence-only baseline, no AUG/near-cognate split. Their pooled nucleotide PR-AUC of 0.767 is
  **not comparable** to any number of ours and should not be quoted as a bar.
- **Condition handling:** a binary stress flag, weaker than RNA conditioning.
- **Benchmark claims:** they do not compare IncepTIS to any other model, and they do not compare
  against Ribo-TISH at all — their method comparison is TISCA2 (the assay) vs ORF-RATER.

---

## 8. What we are borrowing

1. **The learned threshold** (§5). Their binary head is the regression logit minus a learned
   parameter; we need exactly that shape to make the cutoff deployable.
2. **Orthogonal validation.** N-terminal proteomics establishes a start as real because it produced
   a protein with that N-terminus — no Ribo-seq, no caller. Given §17 (83% of reproducible
   later-exon near-cognate sites are in-frame in-CDS, AGG 2.5x enriched, CHX peaks 7x background),
   this is the only clean discriminator between a start and an elongation pause, and it is the one
   control we lack. Public HeLa/K562 N-terminomics suffices; no new sequencing.
3. **AAG/AGG are mostly drug artifacts.** Their inhibitor-free assay finds AAG 0.3% / AGG 0.4% of
   starts; ORF-RATER on harringtonine TI-seq gives 4.6% / 2.4%, and they attribute the difference
   to drug-induced elongation arrest. Our §17 independently found those codons behaving like
   pauses. **These two results corroborate each other** and should be cited as such — it turns
   §17 from an internal caveat into externally supported biology. Consequence for us: tag AAG/AGG
   separately (§3.3) and never count in-CDS in-frame near-cognate sites as plain positives.
4. **Later, the design-and-measure loop.** They designed 5' leaders and validated them with a
   dual-luciferase assay under arsenite. That is the eventual capstone, and we could do it better
   since we predict CHX too — a reporter's elongation profile, not just its initiation. After §4's
   metrics pass.

---

## 9. Corrections to our own docs

Both found while verifying against the fork; both currently misleading.

1. **FINDINGS §15 and Principles §7 mis-scope the Ribo-TISH grouping issue.** As written it reads
   as a general defect in the caller we run and sends the reader to the ribotish repo. It is not:
   the fork reruns all aTIS conditions with byte-identical output. The failure was confined to the
   **chr7 synthetic-BAM evaluation path**, where one transcript can hold a large share of TIS reads.
   Re-scope the note accordingly.
2. **The suggested fix (`while`) does not work.** Verified against the fork's own `pidx` and
   `_est_nb`: `while` removes the `None` entries but fills those slots with the *same* score, so
   `pidx` — which returns the first boundary `>= score` — still never reaches them. Those groups
   receive no transcripts and `_est_nb` raises `ValueError: max() iterable argument is empty`. The
   crash is unchanged. A bare `while` also `IndexError`s, since `parts[-1] == 1` and `s` ends at
   exactly `ctotal`.

   The real behaviour: with the default `--nparts 10`, one top-scoring transcript holding **>~15%**
   of TIS reads leaves at least one group unreachable. A correct fix **deduplicates boundaries and
   merges empty groups** before estimating, accepting fewer groups than `nparts` — ten non-empty
   equal-mass groups do not exist when one transcript holds 45% of the reads. It is a two-file
   change (`src/zbio/ribo.py` plus `src/run/predict.py:161`, which writes `range(len(parts))` rows).
   Not needed if the synthetic-BAM path is retired per §5. Caveat: not reproduced end-to-end on real
   U2OS/RPE1 BAMs, and it would alter background grouping for libraries that currently complete, so
   the fork's `test_exactness.py` / `test_equivalence.py` are the gate.

---

## 10. Phases

| Phase | What | Cost |
|---|---|---|
| 0 | Data contract, truth set, P-site convention assertion, differential-start counts | small, CPU |
| 1 | Re-baseline **class 3** unchanged on clean data; **run `track_tis_eval` on it** (the blank number from §0); rerun the **class-1** baseline on clean labels | 4 folds x 3 seeds, GPU |
| 2 | M1/M2/M3 into the scorecard | small, CPU |
| 3 | Native caller + learned threshold; three-way decomposition | small, fast loop |
| 4 | Architecture work, driven by whatever M1/M2 diagnosis says limits them | the bulk |
| 5 | Proteomics validation | small, needs a search pipeline |
| 6 | Attribution and in-silico mutagenesis, validated first on variants with known effects | moderate |

Phases 0-3 are the commitment. Phase 1's table and Phase 3's transfer check decide whether Phase 4
is needed at all.

---

## 11. Risks

- **§15-16's gains were measured against metrics this spec demotes.** The spliced backbone (+0-7
  points on fine-scale tracks) and flux CHX (uORF framing roughly doubled) may look smaller under
  M1/M2. This is why Phase 1 re-baselines the recipe *unchanged* — it fixes the reference point
  before anything moves.
- **M2 may not be measurable.** If the clean data has too few reproducibly-differential starts
  (§3.5), the differential metric has no substrate and the variant work loses its validation rung.
- **A custom caller can flatter the model.** Arm 2 of the three-way decomposition is the only thing
  preventing that; it is not optional.
- **The native caller breaks comparability with the literature.** Mitigated by keeping Ribo-TISH on
  real BAMs as truth and reporting the caller-free `track_tis_eval` route alongside.
