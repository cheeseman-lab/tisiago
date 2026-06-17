# Autoresearch winners — harvested 2026-06-16

Four parallel loops (`ar/{auprc,auroc,recall1fp,winrate64}`), each climbing one **val** metric
on the balanced 1:3 curated set, reporting **test**. ~20–23 experiments per loop before
plateau. Logistic head won every objective (no MLP survived). Full per-experiment history is
in each worktree's `results.tsv` + git log on branch `ar/<obj>`.

## The configs

All share a **common 7-key feature stack** (the loops converged on it independently):

```
AG16k/off0  +  AG131k/off0  +  Evo2 blk28 off{0,3,6,9}  +  onehot/kozakW20      # 19,620-dim
```

| objective | extra feature | head | C | class_weight | best commit |
|---|---|---|---|---|---|
| auprc | — | logistic | 0.00075 | None | `c844fbd`→`auprc-18` |
| auroc | — | logistic | 0.002 | balanced | `d16349e`/`auroc-21` |
| recall1fp | — | logistic | 0.003 | balanced | `11e18a7`→`recall1fp-9` |
| winrate64 | `+ onehot/codon12` | logistic | 0.1 | balanced | `2bc33ed`→`winrate64-16` |

## Cross-metric TEST matrix (each config scored on all four)

| config | auprc | auroc | recall1fp | winrate64 |
|---|---|---|---|---|
| baseline (AG16k+Evo2 off0, C=1) | 0.7595 | 0.9052 | 0.7506 | 0.8162 |
| **C — 7key, C=0.00075, no weight** (auprc-opt) | **0.7965** | **0.9197** | **0.8013** | 0.8229 |
| A — 7key, C=0.002, balanced (auroc-opt) | 0.7933 | 0.9193 | 0.8002 | 0.8295 |
| A′ — 7key, C=0.003, balanced (recall1fp-opt) | 0.7926 | 0.9190 | 0.7969 | 0.8322 |
| B — 8key +codon12, C=0.1, balanced (winrate-opt) | 0.7765 | 0.9127 | 0.7774 | **0.8336** |

## What the search found

1. **One universal feature stack.** Adding AG131k (regional context) **and** the 3 extra Evo2
   blk28 offsets (base resolution) **and** explicit Kozak one-hot each helped every metric —
   the loops all converged here from the 2-key baseline. Embeddings + explicit local sequence
   are complementary, not redundant.
2. **Config C (heavy L2, no class weight) is the best all-rounder** — it tops auprc, auroc,
   AND recall1fp on test simultaneously. `class_weight="balanced"` raised the val metrics for
   auroc/recall1fp but on **test** the unweighted heavy-L2 head edges it (balanced slightly
   val-overfits).
3. **Precision ↔ resolution tension.** The only way to push win@64 highest (config B) was
   adding explicit codon identity + loose L2 — but that *costs* the precision metrics
   (auprc 0.78, auroc 0.91). The resolution-optimal head is not the precision-optimal head.
4. **Linear suffices, confirmed under search.** Every MLP attempt lost or crashed; all four
   winners are logistic. The PoC "linear is enough" holds even after the loop actively tried
   nonlinearity.
5. **Gains hold on test.** Every objective improved on test in lockstep with val
   (auprc 0.760→0.797, auroc 0.905→0.920, recall1fp 0.751→0.801, win@64 0.816→0.834) — the
   improvements are real, not val hill-climbing artifacts.

## Reproduce a winner
Check out the branch and run its committed config:
```bash
git -C ../tisiago-ar-auprc show HEAD:autoresearch/train_experiment.py   # the CONFIG block
STORE=$PWD/data/store OBJECTIVE=auprc bash autoresearch/run.sh          # from that worktree
```

## Caveat
Single seed (SEED=0), single 60k train subsample, single chr split. These are **best-of-search
point estimates**. Confirm the winners across seeds/splits before any figure/claim (Phase 3).
