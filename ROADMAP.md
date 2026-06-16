# tisiago — roadmap & status

Single source of truth for the work. The north star (see `CLAUDE.md` → *Direction*):
a **general codon → P(initiation) predictor** — any codon in any expressed transcript →
a calibrated probability of being a translation-initiation site, including confidently
**rejecting non-starts**. Build broad first, then narrow.

Headline metrics: (1) recall @ ≤1 false-positive-per-transcript at true imbalance;
(2) non-cognate negative-control score ≈ 0. *Not* the curated-set AUROC.

_Last updated: 2026-06-16._

---

## Status at a glance

| Phase | What | Status | Artifacts |
|---|---|---|---|
| **PoC** | Frozen embeddings rank curated candidates | ✅ done | `eval.py`, `resolution.py`, `FINDINGS.md` |
| **P1** | Calibration machinery + caller metrics | ✅ **done, merged** | `caller.py`; [plan](docs/superpowers/plans/2026-06-16-phase1-caller-calibration.md) |
| **P2** | Global all-codon calibrated caller | 🟡 **code done; GPU scan running** | `enumerate_codons.py`, `scan_eval.py`, `run_tis_scan.sh`; [spec](docs/superpowers/specs/2026-06-16-phase2-global-allcodon-caller-design.md) · [plan](docs/superpowers/plans/2026-06-16-phase2-global-caller.md) |
| **P3** | Autoresearch the head (robustness + CV) | ⬜ **not specced** | — |
| **P4** | TIS efficiency regression (HeLa first) | ⬜ **not specced** | — |

---

## P1 — calibration machinery ✅

`src/tisiago/caller.py`: train on `train`, isotonic-calibrate on held-out `val` (chr7),
report reliability + Brier + recall @ FP-per-transcript budget on `test`. Pure CPU, curated
3:1 store.

**Result:** AUPRC 0.741; the logistic head is *already well-calibrated* (isotonic barely
moves Brier 0.1066→0.1064); **recall 0.733 @ ≤1 FP/transcript**. Because isotonic is
monotonic the recall is calibration-invariant. Honest caveat baked into the output: curated
3:1, not true imbalance.

---

## P2 — global all-codon calibrated caller 🟡

The deliverable. Reuses the pipeline almost wholesale (window/position decoupling).

**Done (CPU, merged):**
- `enumerate_codons.py` — GENCODE v49 GTF → dense scan manifest of *every codon* in
  held-out transcripts (`mrna_index` coords). Coordinate convention pinned by a test against
  the curated manifest (`+`: `coords[i]`; `-`: `coords[i]+1`).
- Real scan manifest produced (test-only): **5.02M positions, 1,443 transcripts, 3,555
  positives** (non-cognate 4.20M / near-cognate 737k / AUG 87k).
- `scan_eval.py` — apply the curated-trained calibrated head to the dense scan store;
  recall @ true imbalance over AUG+near-cognate + non-cognate≈0 grounding.
- `run_tis_scan.sh` — SLURM wrapper (reuses `extract.py`).

**In flight (GPU):** scan extraction over the test-only manifest, headline keys.
- ag16k → A6000 array `10177015` (~30–40 min)
- evo2_8k → A100 array `10177016` (~2.5–3.5 hr; bottleneck — 2-GPU cap, 12 keys)
- assemble → `10177022` (pending `afterok` both) → `data/scan_store/`

**Remaining to close P2:**
- [ ] (early) when ag16k clears: `python -m tisiago.store --manifest data/scan_manifest.parquet --parts-dir data/scan_parts --store-dir data/scan_store_ag` → `scan_eval --scan-store data/scan_store_ag --keys alphagenome_jax/L16k/decoder_1bp/off0.npy` for AG-only numbers.
- [ ] when both clear: combined `scan_store/` assembles → `python -m tisiago.scan_eval --scan-store data/scan_store` for the AG+Evo2 headline.
- [ ] record the true-imbalance recall + non-cognate≈0 result in `FINDINGS.md`.

**Scope decision (logged):** scanning the *whole mature transcript* (5′UTR+CDS+3′UTR),
every frame. Open option to restrict to 5′UTR+CDS (drop 3′UTR, where initiation ≈0) — kept
whole-transcript for the first run for maximal grounding.

**Known size note:** ~57 GB (test-only, AG16k+Evo2). Scan test-only since calibration uses
the curated `val`.

---

## P3 — autoresearch the head ⬜ (not specced)

Use the `autoresearch` skill to autonomously sweep head architectures (logistic → MLP
depth/width, regularization, feature-set & offset combinations) with proper
cross-validation across seeds/splits. Goal: robustness, not a single-seed point estimate.
Subsumes the old "narrow the stats" step.

**Open questions before a spec:** search space bounds; CV scheme (chromosome-fold);
compute budget / autoresearch loop config; what "robust enough" threshold gates success;
whether the evo2 layer/offset sweep (the 12 keys P2 is computing) feeds this directly.

---

## P4 — TIS efficiency regression (HeLa first) ⬜ (not specced)

Move beyond yes/no into quantitative initiation: regress the unused per-condition
translational-efficiency label `max_norm_HeLa` on the frozen embeddings, restricted to
HeLa.

**Open questions before a spec:** which rows (positives only? expressed_HeLa filter?);
target transform (raw `max_norm_HeLa` vs log); metric (Spearman / R²); whether to reuse the
curated store or the dense scan; how the multi-line labels (K562/RPE1/U2OS) factor later.

---

## Immediate next action

Waiting on the GPU scan (~3–4 hr, evo2-bound). First real P2 numbers come from the AG-only
early eval (~30–40 min). Then combined AG+Evo2 when evo2 lands. After P2 closes: spec P3.
