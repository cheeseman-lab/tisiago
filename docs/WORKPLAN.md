# Work plan — AlphaGenome-only cleanup and readiness for the clean data

**Date:** 2026-11-12 · **Status:** proposed, awaiting sign-off on §1 (names) before any file moves.

Goal: by the time the regenerated tracks and the final start set land, the repo holds one AlphaGenome-only
baseline, one track model, one evaluation layer, and the Phase 0/2 code from the
[re-scope spec](superpowers/specs/2026-11-11-p6-rescope-native-caller-design.md), with every name saying what
the thing is. Nothing in this plan submits a Slurm job.

Decisions already taken (2026-11-12): AlphaGenome only, so open decision #1 in
[`P6_PROJECT_PLAN.md`](P6_PROJECT_PLAN.md) resolves to an **AG-only baseline** (recall 0.225, FINDINGS §7), stated
as such wherever the baseline is quoted. Removed code is deleted, not archived in-tree; the pre-cleanup state is
kept as the tag `archive/pre-cleanup-2026-11-12`. On-disk data is inventoried only.

---

## 1. Names

The current names come from the order things were tried (P1-P6, Option B, AR, experiments 1-3, steps 1-2,
classes 1-3, `rnacond`, `sbflux_<g>`). They are retired in code, configs, run IDs and docs. FINDINGS keeps its
§ numbers because issues and specs cite them; a header table maps sections to the names below.

### Lines of work

| New name | Was | What |
|---|---|---|
| **baseline** | class 1, P1/P2/Option B, "frozen embeddings" | Frozen AlphaGenome embeddings + logistic head, one label per codon. The comparator. |
| **floor** | "one-hot ±20 bp floor" | One-hot codon / Kozak ±20 nt features + logistic head. The sequence-only floor every model is judged against. |
| **tracks** | P6, seq2func, class 3 | Fine-tuned AlphaGenome, RNA-conditioned, predicting harringtonine (TIS) and CHX tracks. |
| **calling** | Ribo-TISH on synthetic BAMs, `track_tis_eval` | Turning tracks into start calls: the caller-free score, the native caller (Phase 3), Ribo-TISH on real BAMs as reference. |
| **truth** | (new) | The reproducible-start truth set from the label pipeline, with codon / region / pause tags (Phase 0). |

Removed lines: Evo2 (all), class 2 (per-condition `shared` / `condition:<C>` arms), P4 efficiency regression,
tree heads (XGBoost/LightGBM/RF, which tied logistic in §9), the synthetic-BAM round trip.

### Inside `tracks`: stages, outputs, conditioning

A tracks run is named by four axes, always in this order:

| Axis | Values | Was |
|---|---|---|
| **stage** | `genomic` — fine-tuned on genomic windows, head in genome space · `spliced` — warm-started from `genomic`, fine-tuned on the spliced mRNA, head in transcript space | step 1 / `rnacond` · step 2 / `txtrain` |
| **chx** | `free` — CHX predicted per position · `flux` — CHX = predicted initiation propagated in frame, stopped at stop codons | (default) · `chx_flux` replace mode |
| **cond** | `pos` — positional RNA into the head (today) · `film` — global regulator vector, FiLM, learned · `corgi` — Corgi's pretrained pathway frozen (issue #1) · suffix `@head` / `@trunk` for placement | (only `pos` exists) |
| **holdout** | `K562` · `HeLa` · `U2OS` · `RPE1` (RPE1 async/quiescent/senescent are one group) | `rnacond:<G>`, `<g>` |

Run ID: `tracks.<stage>.<chx>.<cond>.ho-<line>.s<seed>`, e.g. `tracks.spliced.flux.pos.ho-K562.s0` (was
`sbflux_k562_s0`). The **reference recipe** — what Phase 1 re-baselines unchanged — is
`tracks.spliced.flux.pos`.

The fast loop keeps its name (`fastloop`): it is a tool for screening head-side changes on a frozen embedding
cache, not a line of work. Its configs move under `configs/tracks/fastloop/`.

### Baseline arms

`baseline.ag16k` (AlphaGenome 16 kb decoder 1 bp), `baseline.ag131k`, `floor.codon12`, `floor.kozak20`. The
headline comparator is `baseline.ag16k` unless §7's 131k number says otherwise — to check when re-baselining.

### Package layout after the move

```
src/tisiago/
  baseline/    extract, tiling, sequence, extraction_io, store, shard_io, dataset_config, manifest,
               enumerate_codons, backend_contract, caller, head (was dense_caller, logistic only), linear_head
  tracks/      (was seq2func) samples, psite, tracks, targets, data, augment, masking, rna_features,
               expression, model, loss, checkpoint, windows, train (genomic stage), txcache, fastloop,
               txtrain -> spliced, predict_tracks, orf_features, tx_features
  calling/     track_score (was track_tis_eval), native caller (Phase 3), ribotish (parsing real calls)
  truth/       schema + loader + tagging (Phase 0)
  evaluation/  scorecard (M3 guardrails), start_metrics (M1, M2), bootstrap + near-neighbour helpers
```

---

## 2. Cleanup steps, in order

Each step ends with `python -m pytest` green and `ruff check` clean; nothing moves on until it is.

1. **Tag and branch.** Tag `archive/pre-cleanup-2026-11-12` at `b7996fb`, push the tag, branch
   `cleanup/alphagenome-only` from `p6-0-tracks`.
2. **Remove Evo2.** `extract_transcript`, `transcript_inference`, `transcript_projection`, `projected_score`,
   `representation_eval` (after moving `_bootstrap` and `near_neighbor_pairs` into `evaluation/`),
   `store_overlap`, `compare_extractions`; `configs/tis_evo2_*`, `configs/dataset_dense_ag7*` (replaced by an
   AG-only dataset config); `requirements-evo2.lock`, `scripts/run_tis_evo2_*`, `_probe_evo2_layers.py`,
   `run_tis_extract_transcript.sh`, `run_tis_txp_overlap_contract.sh`, `run_tis_representation_eval.sh`,
   `run_tis_dense_representation_eval.sh`; their tests; Evo2 keys in `.env.example`. Gruyerenome's Evo2 support
   is untouched — it lives in that repo.
3. **Remove the other retired baseline code.** `head_xgb` (and the `--head` switch in `dense_caller`),
   `efficiency_head`, `eval.py` + `resolution.py` (PoC, superseded by `caller`), `scan_eval` / `scan_score` /
   `compare_heads` unless the AG-only re-baseline needs dense scoring (check `run_tis_scan.sh` usage first),
   `saerens_control.py`, `build_report.py`, `kozak_onehot_scan.py` if superseded by `floor`.
4. **Remove class 2 and the synthetic-BAM path.** The `shared` and `condition:<C>` arms in `train` /
   `evaluate` / `predict_tracks` / `periodicity`; `synthetic_bam`, `roundtrip`, `predicted_calls`,
   `trial_score`, `run_p6_trial.sh`, `run_p6_roundtrip.sh`, `run_p6_predicted_calls.sh`. `overfit.py` is
   imported by the live trainer only for constants (`HEAD_CHOICES`, `CHX_KINDS`, …) and `finite_or_none`: move
   those into `model` / `checkpoint`, then delete the overfit loop. `periodicity` keeps only the functions
   `scorecard` uses. Check whether the RNA-conditioned genomic stage still needs `evaluate.py`'s per-condition
   code paths before cutting.
5. **Rename.** `seq2func` -> `tracks`, then the moves in §1's layout, then CLI flags and run IDs
   (`--arm rnacond:K562` -> `--holdout K562`; `--stage`, `--chx`, `--cond`). Scripts `run_p6_*.sh` ->
   `run_tracks_*.sh`, `run_tis_*.sh` -> `run_baseline_*.sh`. Old checkpoints carry head names like
   `p6_rnacond_hela` in `run.json`; they are not loaded again (Phase 1 retrains), so no compatibility shim —
   the loader rejects them with a message naming the tag.
6. **Move the reference recipe into configs.** `autoresearch/p6flux/configs/g7_replace_fixed_grad_k1.yaml` ->
   `configs/tracks/spliced_flux_pos.yaml`; the genomic-stage arguments that produced the warm start written out
   as `configs/tracks/genomic_free_pos.yaml` (from the run.json of the §15 checkpoints). After this, the
   reference recipe is reproducible from `configs/` alone.
7. **Remove `autoresearch/`.** The results it holds are already in FINDINGS §6-17; ledgers stay recoverable
   from the tag. Before deleting, grep FINDINGS / PRINCIPLES / specs for `autoresearch/` paths and replace each
   with the tag-qualified path.
8. **Docs.** One entry point per question:
   - `README.md` — what the repo does, the five lines, how to run each. Absorbs `ARCHITECTURE.md`.
   - `docs/P6_PROJECT_PLAN.md` -> `docs/PLAN.md`, absorbs `ROADMAP.md`'s live status; ROADMAP deleted.
   - `docs/P6_PRINCIPLES.md` -> `docs/PRINCIPLES.md`; `INFERENCE_AND_EVALUATION.md` stays, AG-only.
   - `FINDINGS.md` — header table mapping § to the new names; Evo2 numbers stay as history, marked so.
   - Delete `HANDOFF_OPTION_B.md`, `docs/tisiago_report.html`, `.superpowers/sdd/briefs/`, and the
     superpowers specs/plans except the re-scope spec (moved to `docs/specs/`).
   - `CLAUDE.md` rewritten against the new layout.
9. **Disk inventory** (no deletion): sizes of `data/`, stores, checkpoints, caches, `logs/`, `.venv/`, with
   each item marked keep / superseded, for review.

---

## 3. Readiness for the clean data (built today, run tomorrow)

Written against synthetic fixtures, tested, not run on real data.

1. **`truth/`** — schema and loader for the reproducible-start set: one row per start with transcript,
   genomic and transcript coordinate, codon, `codon_class` (AUG / AAG / AGG / other near-cognate / non-cognate),
   `region` (5'UTR / annotated / in-CDS in-frame / in-CDS out-of-frame / other), `pause` flag, per-replicate
   support, per-line presence. Region and codon tags computed here from the annotation, not trusted from the
   input. Validation reports counts per class per line (Phase 0 "done when").
2. **P-site convention check.** One function that reads a track at a set of known annotated AUG starts on
   each strand and reports the offset of the peak; a test that fails if model tracks and release bigWigs differ
   on the minus strand. Run tomorrow on the first release bigWig — this is Phase 0 item 4.
3. **`evaluation/start_metrics`** — M1 (precision / recall against truth at a fixed calls-per-transcript
   budget, split by `codon_class` and in-CDS vs outside) and M2 (direction accuracy on reproducibly
   differential starts, with the count of such starts reported first). M3 stays in `scorecard`.
4. **Differential-start counter** — per condition pair, the number of starts whose share reproducibly differs
   (Phase 0 item 5).
5. **Data contract** — `docs/DATA_CONTRACT.md`: the files the label pipeline must hand over (per-replicate and
   pooled tracks, matched RNA, truth set, splits), with the fields above. Unknowns about the hand-off format
   listed explicitly, to settle when it lands.

---

## 4. Tomorrow

Data lands -> load into `truth/`, record counts -> P-site check on the first bigWig -> differential-start
counts -> Phase 0 done. Then Phase 1: re-baseline `baseline.ag16k` and `tracks.spliced.flux.pos` on clean
labels, and score `calling.track_score` on the tracks model for the first time.
