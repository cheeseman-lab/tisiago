# Work plan — AlphaGenome-only cleanup and readiness for the clean data

**Date:** 2026-11-12 · **Status:** cleanup done on branch `cleanup/alphagenome-only`; clean-data work starts
2026-11-13. Names are defined in [`PLAN.md`](PLAN.md) §1. The pre-cleanup tree is tag
`archive/pre-cleanup-2026-11-12`.

---

## 1. What changed

| Step | Result |
|---|---|
| Evo2 removed | Extraction, transcript-input (TXP) and projection code, configs, env lock, scripts. Baseline evaluator compares `baseline.ag` against `floor.kozak20` / `floor.codon12`. |
| Retired baseline code removed | Tree heads, P4 efficiency regression, curated-vs-dense caller, PoC eval/resolution, head comparison, report, Option B handoff. |
| Synthetic-BAM path removed | `roundtrip`, `predicted_calls`, trial scorer, overfit gate. `synthetic_bam` kept as a test helper for P-site tests. |
| Packages renamed | `seq2func` -> `tracks` (`train` -> `train_genomic`, `txtrain` -> `train_spliced`, `tracks` -> `track_io`); `calling/` (`track_score`, `ribotish`); `baseline/` (every frozen-embedding module; `representation_eval` -> `evaluate`). Scripts `run_tracks_*`, `run_calling_*`, `run_baseline_*`. Default data root `data/p6` -> `data/tracks`. |
| Reference recipe in configs | `configs/tracks/genomic_fc2d.args` (stage 1, from the §15 run.json) and `configs/tracks/spliced_flux_pos.yaml` (stage 2, the §16 config; resolved config identical apart from name and cache path). `train_spliced` now defaults to it (was the §15 head `p3r02_onehot41_l3`). |
| Tools moved out of `autoresearch/` | `baseline.onehot` (floor features), `tracks.cache_seq` (mRNA sequence into transcript caches, needed by the ±20 nt one-hot branch). |
| `autoresearch/`, `.superpowers/`, historical specs/plans, ROADMAP, ARCHITECTURE deleted | Recoverable from the tag. Re-scope spec kept at `specs/2026-11-11-rescope-native-caller.md`. |
| New: Phase 0/2 code | `truth/` (schema, release-table loader, P-site check, differential starts), `evaluation/start_metrics` (M1, M2), `DATA_CONTRACT.md`. 65 tests, synthetic data only. |

Tests: CPU suite (`tisiago` env) and JAX suite (`agft` env, on a CPU node) green; the JAX suite matched the
pre-cleanup tag before the rename (the only errors are pysam imports missing in `agft`, identical in both).

## 2. Deliberately not done (debt)

- **Per-condition arms** (`shared`, `condition:<C>`) still live in `train_genomic`, `tracks.evaluate`,
  `predict_tracks`, `periodicity`, and `model` keeps the explored head kinds (`mlp` ... `rnacond_fc2`). They are
  woven through ~40 tests and the live code paths; remove after Phase 1 has re-trained the reference recipe on GPU,
  so numeric identity can be checked rather than assumed.
- **Registered head names** inside checkpoints stay `p6_rnacond_<line>`: renaming them breaks loading of existing
  checkpoints. Internal identifier only.
- **CLI flags** keep `--arm rnacond:<line>`; the run-ID axes (`stage`, `chx`, `cond`, `holdout`) are encoded in
  `--out` names for now.
- `txcache` defaults still name an old run (`ar_rnacond_k562_v2`); always pass `--name` / `--checkpoint`.

## 3. Tomorrow (Phase 0, then Phase 1)

1. Load the release into `truth.load_truth`; write `summarize_truth` counts per line x codon class x region.
2. P-site check (`truth.psite_check`) on one release bigWig per strand vs the model's track convention. The fork
   corrects the 0.2.7 minus-strand +1 (`make_tracks.psite_coord`); expect a 1-nt minus shift until reconciled.
3. Differential-start counts per condition pair (`truth.count_differential`) — decides whether M2 is measurable.
4. Settle the open hand-off questions below with the label-pipeline owner.
5. Phase 1: rebuild targets/windows on the clean tracks, re-run `tracks.spliced.flux.pos` (4 folds x 3 seeds),
   `calling.track_score` on it, and `baseline.ag` + floors on the clean labels.

### Open hand-off questions (DATA_CONTRACT §6)

1. Truth = raw `starts` with replicate flags, or the cross-line-filtered strong set (and does it keep rep1/rep2)?
2. Is `rep{1,2}_called` (full cascade per replicate) the reproducibility bar?
3. No `chx_pause_*` in v50: pause flag stays NA until a source is chosen.
4. In-CDS by the row transcript's CDS, the gene's canonical CDS (`RecatTISType`), or MANE (§17)?
5. Release P-site bigWigs are CPM; counts via `total_psites` in `<C>.tracks.json`.
6. Release RNA is pooled full-read coverage (CPM); the model uses per-library 5'-end counts.
7. Minus-strand convention (item 2 above).
8. Chromosome naming and GENCODE version must match the transcript cache.
9. iPSC / CM have no replicates — excluded from truth?
10. M2 shares: derived here from replicate TIS tracks or supplied; per gene or per transcript?

## 4. Disk inventory (nothing deleted)

| Path | Size | Status |
|---|---|---|
| `data/p6/runs` | 715G | pre-cleanup checkpoints; keep `ar_rnacond_*_fc2d_l2w300` (stage-1 warm starts of §15-16); rest superseded |
| `data/p6/txtrain` | 303G | §15-17 stage-2 runs; keep `sbflux_*` (reference) |
| `data/p6/txcache` | 224G | `fc2d_*` caches for the reference; rebuilt on clean data |
| `data/p6/roundtrip*` | 33G | synthetic-BAM round trip — superseded |
| `data/p6/fastloop` | 13G | fast-loop runs — superseded |
| `data/scan_parts_allsplits` | 1.8T | dense scan shards, AG + Evo2 keys — Evo2 keys superseded |
| `data/scan_store_allsplits` | 361G | dense scan store — superseded by Regime A stores |
| `data/dense_exp_store` | 292G | baseline experiment store (AG + Evo2 keys) — Evo2 arrays superseded |
| `data/dense_txp_parts`, `data/txp_parts`, `data/*txp_representation` | 108G | Evo2 TXP — superseded |
| `data/scan_parts` | 50G | earlier scan shards — superseded |
| `data/store` | 25G | curated store (AG + Evo2) — Evo2 arrays superseded |
| `data/parts` | 15G | curated extraction shards — superseded |
| `data/preds_*.npz`, `data/head_comparison.tsv` | 230M | tree-head comparison — superseded |
| `.venv/evo2-next` | 6.2G | Evo2 env — superseded |
| `$TISIAGO_DENSE_STORE` (`.../tisiago_store/dense_ag7`) | not sized | SSD baseline store; AG keys reusable until clean labels change rows |

Clean labels change the manifest, so every store and cache above is rebuilt in Phase 1; the inventory is for a
deletion decision, not a dependency.
