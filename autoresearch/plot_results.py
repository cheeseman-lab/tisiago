"""Quick plots of the curated 3:1 autoresearch fleet (4 objectives, 4 worktrees)."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OBJS = ["auprc", "auroc", "recall1fp", "winrate64"]
ROOT = Path("..")
OUT = Path("autoresearch/plots")
OUT.mkdir(parents=True, exist_ok=True)


def load(o):
    df = pd.read_csv(ROOT / f"tisiago-ar-{o}" / "autoresearch" / "results.tsv",
                     sep="\t", encoding="utf-8", encoding_errors="replace")
    for c in df.columns:
        if c.endswith("_val") or c.endswith("_test"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


data = {o: load(o) for o in OBJS}
base = data["auprc"][data["auprc"].status == "baseline"]


def baseval(substr, col):
    m = base[base.description.str.contains(substr, case=False, na=False)]
    return float(m[col].iloc[0]) if len(m) else np.nan


# ---- Fig 1: optimization trajectories (val) -------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, o in zip(axes.ravel(), OBJS):
    df = data[o]
    exp = df[df.status.isin(["keep", "discard", "crash"])].reset_index(drop=True)
    col = f"{o}_val"
    x = np.arange(1, len(exp) + 1)
    y = exp[col].values
    kept, disc = exp.status.values == "keep", exp.status.values == "discard"
    ax.scatter(x[disc], y[disc], c="lightgray", s=25, label="discard")
    bsf = np.fmax.accumulate(np.nan_to_num(y, nan=-np.inf))
    bsf[~np.isfinite(bsf)] = np.nan
    ax.plot(x, bsf, c="tab:green", lw=1.5, alpha=0.6, label="best-so-far")
    ax.scatter(x[kept], y[kept], c="tab:green", s=40, zorder=3, label="keep")
    for sub, c, ls in [("AG16k", "tab:blue", "--"), ("kozak", "tab:orange", ":")]:
        v = baseval(sub, col)
        ax.axhline(v, ls=ls, c=c, lw=1.2,
                   label=("AG+Evo2" if sub == "AG16k" else "Kozak floor") + f" {v:.3f}")
    ax.set_title(f"climbing {o}_val  (n={len(exp)})")
    ax.set_xlabel("experiment #")
    ax.set_ylabel(f"{o}_val")
    ax.legend(fontsize=7, loc="lower right")
fig.suptitle("Curated 3:1 autoresearch — optimization trajectories (val)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "trajectories.png", dpi=130)

# ---- Fig 2: AR best (test) vs floors, per objective -----------------------------------
fig2, ax = plt.subplots(figsize=(10, 5))
groups = [("codon", "tab:gray", lambda m: baseval("codon", f"{m}_test")),
          ("Kozak", "tab:orange", lambda m: baseval("kozak", f"{m}_test")),
          ("AG+Evo2 base", "tab:blue", lambda m: baseval("AG16k", f"{m}_test")),
          ("AR best", "tab:green",
           lambda m: float(data[m][data[m].status == "keep"][f"{m}_test"].max()))]
w = 0.2
for i, (lab, c, fn) in enumerate(groups):
    vals = [fn(m) for m in OBJS]
    bars = ax.bar(np.arange(len(OBJS)) + i * w, vals, w, label=lab, color=c)
    ax.bar_label(bars, fmt="%.2f", fontsize=7)
ax.set_xticks(np.arange(len(OBJS)) + 1.5 * w)
ax.set_xticklabels(OBJS)
ax.set_ylabel("test metric")
ax.set_ylim(0, 1)
ax.set_title("Autoresearch best (test) vs floors, per objective")
ax.legend(ncol=4, fontsize=9)
fig2.tight_layout()
fig2.savefig(OUT / "best_vs_floor.png", dpi=130)
print("wrote", OUT / "trajectories.png", "and", OUT / "best_vs_floor.png")
