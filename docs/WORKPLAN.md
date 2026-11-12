# Work plan — AlphaGenome-only cleanup and readiness for the clean data

**Date:** 2026-11-12 · **Status:** cleanup merged to `main` (PR #2); clean-data work starts 2026-11-13. Names
are defined in [`PLAN.md`](PLAN.md) §1.

**Archive tags** (all on GitHub; `main` is the only branch):

| Tag | Holds |
|---|---|
| `archive/pre-cleanup-2026-11-12` | The whole pre-cleanup tree: Evo2, retired baseline heads, synthetic-BAM path, `autoresearch/` (every loop's program, results and ledger) |
| `archive/ar-auprc`, `archive/ar-auroc`, `archive/ar-recall1fp`, `archive/ar-winrate64` | The four baseline autoresearch loops, one commit per iteration |

The loops' untracked run logs (`agent.log`, `loop.log`, `run.log`, `preds.npz`) are on disk only, in
`data/archive/autoresearch_ar_logs/<objective>/`.

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

## 2. Debt and deliberate choices

- **Per-condition arms and explored head kinds: removed after review (2026-11-12).** The only genomic arm is
  `rnacond:<line>` and the only registered head is `rnacond_fc2d`; `tx_features`, `orf_features` and the
  periodicity CLI went with them. `rnacond_fc2d` parameters (48 leaves), forward output and gradients were checked
  byte-identical against the archive tag on CPU. Phase 1 still re-trains on GPU before anything is compared.
- **Registered head names** inside checkpoints stay `p6_rnacond_<line>`: renaming them breaks loading of existing
  checkpoints. Internal identifier only.
- **CLI flags** keep `--arm rnacond:<line>`; the run-ID axes (`stage`, `chx`, `cond`, `holdout`) are encoded in
  `--out` names for now.
- `txcache` has no default run: pass `--name` (or `--cache`) and `--checkpoint`.

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

## 4. Disk (superseded data deleted 2026-11-12, about 3 TB)

Deleted: everything this section listed as superseded — the round-trip and fast-loop runs, `scan_store_allsplits`,
`scan_parts`, `parts`, all Evo2 TXP folders, the Evo2 arrays in the mixed stores, the tree-head predictions and
`.venv/evo2-next`. Not recoverable; the code that made them is in the archive tags.

What remains:

| Path | Size | Holds |
|---|---|---|
| `data/p6/runs` | 81G | `ar_rnacond_{hela,k562,rpe1,u2os}_fc2d_l2w300` (stage-1 warm starts of the reference) and their `_s1` seeds; 85 other checkpoints deleted |
| `data/p6/txtrain` | 93G | `sbflux_*` (the 12 reference runs) and their exports; results `.tsv` files; other stage-2 runs deleted |
| `data/p6/txcache` | 224G | `fc2d_*` caches for the reference; `ar_rnacond_k562_v2` cache |
| `data/scan_parts_allsplits` | 299G | AG 16 kb / 131 kb dense scan shards (Evo2 shards deleted) |
| `data/dense_exp_store` | 27G | baseline experiment store, AG + one-hot (Evo2 deleted) |
| `data/store` | 1.2G | curated store, AG + one-hot (Evo2 deleted) |
| `data/archive/` | small | run logs of the baseline autoresearch loops (§ archive tags above) |
| `$TISIAGO_DENSE_STORE` (`.../tisiago_store/dense_ag7`) | not sized | SSD baseline store; AG keys reusable until clean labels change rows |

Not reviewed, left in place: `data/p6flux`, `data/p6nc`, `data/p6nclater`, `data/p6sharp`, `data/p6utr5`,
`data/scan_parts_ag`, `data/scan_store_ag` and the other `data/p6/` subfolders (`windows`, `expression`, `eval`, ...).

Clean labels change the manifest, so every store and cache above is rebuilt in Phase 1; what remains is kept for
reference, not as a dependency.
