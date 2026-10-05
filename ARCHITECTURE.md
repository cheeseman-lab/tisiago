# tisiago — Architecture

The **TIS-prediction orchestrator**: reads a candidate table, embeds each candidate
codon through a frozen genome foundation model, persists a row-aligned vector store,
and trains lightweight heads. The whole repo exists to test one claim — *a frozen
embedding carries a translation-initiation signal that no foundation model is trained
to emit* — as cheaply as possible.

---

## 1. Where tisiago sits (the three-repo pipeline)

```
┌───────────────┐   manifest.parquet   ┌─────────────────────────────────┐   embeddings   ┌──────────────┐
│ swissisoform  │  ─────────────────▶  │            tisiago              │  ◀──────────▶  │ gruyerenome  │
│  makes table  │   1 row / candidate  │        (this repo)              │  embed_positions│  AG / Evo2   │
└───────────────┘                      └─────────────────────────────────┘                └──────────────┘
   genome biology      labels, splits,        tiling · extract · store · heads               generic
   + Ribo-seq calls    region classes         (no model internals here)                       embedder
```

- **swissisoform** — owns the biology. Emits one row per candidate translation-initiation
  codon: positives (Ribo-seq–called TIS) + matched in-transcript negatives, with labels,
  region class, expression flags, and the chromosome split.
- **gruyerenome** — owns the models. Exposes a generic `embed_positions(sequences, positions)`
  over AlphaGenome / Evo2. Knows nothing about TIS.
- **tisiago** — owns the *orchestration*. The only place that knows both "what a candidate is"
  and "how to turn it into a stored vector." Knows no model internals and defines no biology.

**Design invariant — decouple the forward-pass window from what is persisted.** Embed over a
generous genomic window for context, but store only the single vector at each candidate codon.
Every candidate (positive and negative) is centered identically, so absolute position carries
no signal — the head must discriminate on learned context alone.

---

## 2. Module map

Core path below; the full module list (TXP extraction, contracts, clean multi-seed
evaluation, partial-logit scoring) is in `CLAUDE.md` → Code Structure.

```
src/tisiago/
├── tiling.py      pure geometry      candidate codon ──▶ (tile_start, within-tile offset)
├── extract.py     GPU driver         tiles ──▶ genome fetch ──▶ embed_positions ──▶ .npz shard parts
├── store.py       assembler          .npz parts ──▶ row-aligned [N, D] .npy feature arrays
├── eval.py        head (CPU)         .npy ──▶ logistic/MLP ──▶ AUROC/AUPRC, stratified by start type
├── resolution.py  head (CPU)         .npy ──▶ logistic ──▶ near-neighbour win-rate (base resolution)
├── caller.py      head (CPU)         .npy ──▶ calibrated p ──▶ reliability · recall @ FP/transcript budget
├── enumerate_codons.py  scan setup (CPU)   GTF + genome ──▶ dense scan manifest (every codon)
├── scan_eval.py         global caller (CPU) scan store ──▶ recall @ true imbalance · non-cognate≈0
└── dense_caller.py      Option B (CPU)      curated- or dense-trained heads ──▶ chunked score of dense scan @ true imbalance
```

| Module | Purpose | Depends on | Testable as |
|---|---|---|---|
| `tiling.py` | Grid-snap each candidate's codon-A into a shared forward-pass window | nothing (stdlib) | pure CPU unit — deterministic geometry |
| `extract.py` | Fetch windows, run one forward pass per tile, slice candidate vectors | `tiling`, `gruyerenome`, `pyfaidx` | integration (needs GPU + genome) |
| `store.py` | Scatter shard parts into row-aligned arrays + provenance | numpy/pandas/yaml | integration (needs parts) |
| `eval.py` | Which embeddings carry signal; is it the *interesting* (non-canonical) kind | numpy/pandas/sklearn | runs on store, no GPU |
| `resolution.py` | Is the signal at true single-nucleotide resolution | numpy/pandas/sklearn | runs on store, no GPU |
| `caller.py` | Calibrated probability + caller-shaped metrics (deliverable track) | numpy/pandas/sklearn | runs on store, no GPU |
| `enumerate_codons.py` | GTF→dense scan manifest of every codon (Phase 2) | numpy/pandas/pyfaidx | pure fns unit-tested; coords checked vs manifest |
| `scan_eval.py` | Apply one calibrated head to the dense scan store (Phase 2) | numpy/pandas, `caller` | logic unit-tested; numbers need scan store |
| `dense_caller.py` | Option B: two curated-trained heads (`None` vs `balanced`), chunked memmap scoring of the dense scan @ true imbalance + 3-gene demo; `--features ag` (AG+Kozak, ready first) vs `ag7` (+Evo2, later) | numpy/pandas, `caller`, `scan_eval` | `train_heads` smoke-validated; eval needs scan store |

The cut that matters: **`tiling` is pure and `extract` is a thin driver around it.** All the
subtle coordinate logic (strand, revcomp orientation, grid snap, edge safety) lives in the
pure module that can be tested without a GPU; `extract` only does I/O and batching.

**Two evaluation tracks:** `eval.py` / `resolution.py` measure *ranking* (AUROC,
win-rate) — the PoC sanity check; `caller.py` measures the *deliverable* — a calibrated
probability and recall at a false-positives-per-transcript budget. Phase 2's dense scan
reuses `caller.py`'s metric functions unchanged.

---

## 3. Data flow (end to end)

```
                         ┌─────────────────────── GPU stage (alphagenome / evo2 env) ───────────────────────┐
 manifest.parquet        │                                                                                  │
 192,072 candidate ──────┼──▶ shard by transcript_id (crc32)   ──▶  group_into_tiles(spec)                  │
 rows (row_idx 0..N-1)   │      keeps a transcript's tiles intact      nearby candidates collapse onto      │
                         │      within one shard                       one tile (share factor > 1)          │
                         │                                                  │                               │
                         │                                                  ▼                               │
                         │      _fetch_window (pyfaidx, N-pad, revcomp)  per-tile genomic sequence          │
                         │                                                  │                               │
                         │                                                  ▼                               │
                         │      gruyerenome backend.embed_positions(seqs, positions)   one fwd pass / tile  │
                         │                                                  │                               │
                         │                                                  ▼                               │
                         │      slice candidate (+offset) vectors  ──▶  data/parts/<spec>_shard<NNNNN>.npz  │
                         └──────────────────────────────────────────────────┬───────────────────────────────┘
                                                                             │  (one .npz per shard × spec)
                         ┌─────────────────── CPU stage (any env) ───────────▼───────────────────────────────┐
                         │   store.py: scatter every part's vectors into global [N, D] fp16 arrays           │
                         │   keyed backend::length::layer::offset, row-aligned to row_idx + coverage check   │
                         │                                                  │                                 │
                         │                                                  ▼                                 │
                         │                                          data/store/embeddings/**.npy             │
                         │                                                  │                                 │
                         │   eval.py / resolution.py: load .npy, concat axis=1, filter rows by split column  │
                         │   StandardScaler ─▶ LogisticRegression/MLP ─▶ AUROC · AUPRC · win@Dbp             │
                         └───────────────────────────────────────────────────────────────────────────────────┘
```

**Sharding & tile sharing.** Candidates are sharded by `transcript_id` (stable crc32) so a
transcript's candidates — and therefore its shareable tiles — always land in the same shard.
Within a shard, `group_into_tiles` dedups by `(chrom, tile_start, strand)`: candidates within
`step` bp snap onto one tile and ride a single forward pass (the `share` factor printed per shard).

---

## 4. The tiling idea (why it's the heart of the repo)

```
transcript ─────●────────●──●───────────────────●──────────────▶  (candidate codons, plus strand)
                 \        \ /                    /
   grid (step bp)  ●        ●        ●        ●        ●     round(a / step) * step
                   │        │                  │
                   ▼        ▼                  ▼
   tiles      [── W ──]  [── W ──]          [── W ──]      ◀ two close candidates share one tile

   centered (AlphaGenome, bidirectional):  candidate offset ∈ [W/4, 3W/4]
   left_heavy (Evo2, causal):              candidate ~up_anchor into the fed 5'→3' sequence
```

`MODEL_SPECS` in `tiling.py` is the single source of truth for window geometry:

| spec | backend | window | step | placement | offsets | length tag |
|---|---|---|---|---|---|---|
| `ag16k` | alphagenome_jax | 16,384 | 8,192 | centered | [0] | L16k |
| `ag131k` | alphagenome_jax | 131,072 | 65,536 | centered | [0] | L131k |
| `evo2_8k` | evo2 | 8,192 | 2,048 | left_heavy (up_anchor 5120) | [0,3,6,9] | W8k |

Offsets exist because Evo2 is **causal**: the codon vector can't see the downstream
Kozak/frame positions, so extract also stores vectors a few bp downstream ({0,3,6,9}).
Minus-strand tiles are reverse-complemented at fetch; offsets are computed in the fed
(5′→3′) orientation so the same slicing code works for both strands.

---

## 5. Store layout (the contract between GPU and CPU stages)

Row `i` is always the candidate with `row_idx == i` — the store is row-aligned to the manifest,
so a head experiment never has to join, only index.

```
data/store/
├── manifest.parquet                                  # copy of the candidate table (44 cols)
├── config.yaml                                        # provenance: per-key dim + coverage/N
└── embeddings/
    ├── alphagenome_jax/L16k/decoder_1bp/off0.npy      # [N, 1536] fp16
    ├── alphagenome_jax/L131k/decoder_1bp/off0.npy     # [N, 1536] fp16
    ├── evo2/W8k/blocks.{24,26,28}.mlp.l3/off{0,3,6,9}.npy   # [N, 4096] fp16
    └── onehot/{codon12,kozakW20}.npy                  # [N,12] / [N,164] fp16  (grounding floors)
```

The `onehot/` arrays are raw one-hot sequence (codon, and ±20 bp window) written by
`autoresearch/make_onehot.py` — *not* model embeddings, but loaded through the same key
mechanism so a head can use them as baselines or concatenate them with embeddings.

Key grammar: `backend :: length_tag :: layer :: offset`. A head experiment is therefore just
*"load these few `.npy`, `np.concatenate(axis=1)`, filter rows by `manifest.split`"* — no GPU,
no model, no coordinate logic. That is the payoff of persisting only the per-candidate vector.

**Phase 2 dense scan store.** The global caller reuses this exact layout for a parallel
`data/scan_store*/` (+ `data/scan_manifest*.parquet`, `data/scan_parts*/`), row-aligned to the
scan manifest instead of the curated one — *every* codon in held-out transcripts rather than
curated candidates. Same key grammar, same `.npy` arrays, so the callers load it identically.

**Dense genome-wide data is NOT assembled into a row-aligned monolith.** At ~62 M codons the
full `[N, D]` scatter does not fit in RAM or SSD. Two regimes handle it instead (decision doc
`docs/superpowers/specs/2026-06-18-embedding-dataset-architecture-decision.md`): Regime A
(`scripts/build_store.py`) gathers only the candidate rows into a compact SSD store; Regime B
(`tisiago.scan_score`) streams shards directly to score every codon without materialising the
matrix. Curated-candidate stores (≪ 10 M rows) continue to use `store.py` unchanged.

**Option B (2026-06-17) first used this store as an evaluation substrate**
(`src/tisiago/dense_caller.py`). It later moved to `--train dense` (training on the compact dense
store at 49:1, FINDINGS §7), and the clean multi-seed protocol lives in `representation_eval.py`
(§10). The original design trained the autoresearch-winner 7-key head on the **curated** store (two variants —
`class_weight=None` vs `balanced`), calibrates on curated val, and scores both on the dense
scan TEST split at true ~230:1 imbalance (recall @ ≤1 FP/tx over cognate codons +
non-cognate≈0 grounding + a 3-gene out-of-sample demo). At 19.6k-dim × 5 M test rows (~196 GB
fp16) it never materializes the matrix — it **predicts chunked over a memmap'd store**. The
dense scan is thus an *evaluation* substrate; the head still trains on curated, so it doesn't
re-introduce the train/eval-negative-distribution tangle that parked dense *training*.

The all-splits scan (`scan_manifest_allsplits.parquet`, 62.7 M codons → `scan_store_allsplits/`)
is large (~2.5 TB at the 7-key stack). The earlier **evo2 RAM-bound OOM** (`extract.py`
accumulates all sliced vectors before writing → 12 keys ≈ 85 GB/shard) is sidestepped by the
**blk28-only config** (`tis_evo2_8k_blk28.yaml`, 4 keys) at `--mem=192G`. Streaming parts to
disk remains logged debt for a full 12-key dense run.

---

## 6. Execution & environments

```
scripts/run_tis_pipeline.sh  N
        │
        ├─ run_tis_extract.sh   (SLURM array, GPU)  ── 3 specs × shards ──▶ data/parts/*.npz
        │        env: conda alphagenome / uv .venv/evo2-next   ·   extract.py
        │
        └─ run_tis_assemble.sh  (CPU)  ── store.py ──▶ data/store/
                 env: tisiago

eval / resolution / caller  (CPU, tisiago env)  ── read-only over data/store/

# Phase 2 / Option B — dense scan as a true-imbalance evaluation substrate:
enumerate_codons.py  (CPU)  ── GENCODE GTF + genome ──▶ data/scan_manifest_allsplits.parquet (62.7M codons)
        │
        ├─ run_tis_scan.sh  (SLURM array, GPU)  ── extract.py ──▶ data/scan_parts_allsplits/*.npz
        │        env: conda alphagenome / uv .venv/evo2-next   ·   evo2 blk28-only config @ --mem=192G
        ├─ kozak_onehot_scan.py  (CPU)  ── genome ──▶ kozakW20_allsplits.npy (validated vs curated)
        │
        └─ store.py  (CPU)  ──▶ data/scan_store_allsplits/
                 └─ dense_caller.py  (CPU)  ── curated-trained heads, chunked memmap scoring ──▶ recall @ ~230:1 · non-cognate≈0 · 3-gene demo

scan_eval  (CPU, tisiago env)  ── curated-trained calibrated head applied to data/scan_store/
```

| Stage | Environment | Needs | Entry point |
|---|---|---|---|
| Extraction | conda `alphagenome` / uv `.venv/evo2-next` | tisiago + gruyerenome (editable) + pyfaidx, GPU | `extract.py` via SLURM array |
| Assembly | any (`tisiago`) | numpy/pandas/pyyaml | `store.py` |
| Eval / training | `tisiago` | numpy/pandas/sklearn, no GPU | `eval.py`, `resolution.py`, `caller.py` |
| Scan enumeration | `tisiago` (+pyfaidx) | GTF + genome, no GPU | `enumerate_codons.py` |
| Global caller | `tisiago` | numpy/pandas, `caller` | `scan_eval.py` |

`configs/*.yaml` are **gruyerenome** backend configs (model, batch size); `extract.py` asserts
`config.model == spec.backend` so a mismatched config/spec pair fails fast.

---

## 7. Boundaries at a glance

- **Biology lives upstream** (swissisoform): labels, splits, region classes, expression. tisiago
  never decides what a positive is.
- **Model internals live to the side** (gruyerenome): which layers, how to run AG/Evo2. tisiago
  only names keys and asks for positions.
- **tisiago owns the join**: candidate ↔ window geometry ↔ stored vector ↔ row alignment. Everything
  it adds is in service of making a head experiment a GPU-free `np.load` + `concatenate` + `split` filter.

---

## 8. Autoresearch harness (`autoresearch/`)

A self-contained experiment loop that exploits the GPU-free head substrate: pick a feature
subset, train a light head, score it — in <2 min CPU. Ran as a **4-way parallel fleet**, one
loop per objective metric (**done & harvested 2026-06-16** — see `winners.md`).

```
autoresearch/
├── make_onehot.py        CPU prep   manifest + genome ──▶ onehot/{codon12,kozakW20}.npy (grounding floors)
├── train_experiment.py   MUTABLE    CONFIG (features·head·class_weight·neg_subsample) ──▶ preds.npz
├── evaluate.py           FIXED      preds.npz ──▶ auprc·auroc·recall1fp·winrate64 (val+test); echoes OBJECTIVE
├── run.sh                srun CPU   one iteration: train ──▶ evaluate
├── launch_loop.sh        DRIVER     outer loop: fresh `claude -p` proposes ONE CONFIG ──▶ run ──▶ keep/discard via git
├── program.md            the agent's brief (levers, floors, climb-val-report-test rule)
├── results.tsv           per-worktree keep/discard log
└── winners.md            harvested best config per objective (the deliverable)
```

The cut mirrors the rest of the repo: **`train_experiment.py` is the only mutable surface**
(the autoresearch agent edits its `CONFIG`), `evaluate.py` is the fixed metric (reuses
`caller.recall_at_fp_budget` + `resolution.py`'s win-rate pairing). `launch_loop.sh` drives
the deterministic outer loop so each `claude -p` starts fresh-context and only proposes a
CONFIG edit. Four loops ran in isolated **git worktrees** (`../tisiago-ar-<obj>`, one objective
each), all reading one shared absolute `--store`. Guardrail: every loop **climbs a val metric,
reports test** — no test-set selection. Outcome: logistic on a 19.6k-dim AG+Evo2+Kozak stack
beat the 2-key baseline on every metric (`winners.md`).
```
