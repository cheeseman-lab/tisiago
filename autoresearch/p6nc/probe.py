"""Q1 information content: logistic probes for K562 near-cognate (and ATG) starts on val chr7.

Usage:
    python autoresearch/p6nc/probe.py --data data/p6nc --out data/p6nc/probe.tsv

Sets (rows of ``codons.parquet`` that join the dense store, so every feature exists for every row):
  nc_global   real K562 NC starts vs unused codons of the same identity (no filtered call of any
              condition within +-30 nt), matched on region (5'UTR/CDS/3'UTR) and transcript RNA
              decile, 5 negatives per positive;
  atg_global  the same for ATG starts;
  nc_local    real NC starts vs every uncalled cognate codon within +-30 nt of one (placement);
  ncrep_*     the same two NC sets restricted to reproducible starts (rep1 and rep2 tracks both
              peak within 1 nt of the start).
Each feature set: StandardScaler + L2 logistic, C picked by inner 3-fold grouped CV, outer 5-fold
CV grouped by transcript; out-of-fold AUROC / AUPRC, and for *_local sets the per-site win rate (share
of a positive's local negatives it outranks). Raw (unfitted) scores of the final model are also
reported. Labels are used for evaluation only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NC = ("CTG", "GTG", "TTG", "ACG", "AGG", "AAG", "ATT", "ATC", "ATA")
NEG_PER_POS = 5
LOCAL = 30
BASE = "ACGT"
DENSE_KEYS = {
    "ag16k_store": "embeddings/alphagenome_jax/L16k/decoder_1bp/off0.npy",
    "ag131k_store": "embeddings/alphagenome_jax/L131k/decoder_1bp/off0.npy",
    "evo2_w8k": "embeddings/evo2/W8k/blocks.28.mlp.l3/off0.npy",
    "evo2_txp": "embeddings/evo2/TXP/blocks.28.mlp.l3/off0.npy",
    "kozakW20_store": "embeddings/onehot/kozakW20.npy",
}


def onehot(ctx: pd.Series) -> np.ndarray:
    arr = np.frombuffer("".join(ctx).encode(), dtype="S1").reshape(len(ctx), -1)
    return np.concatenate([(arr == b.encode()).astype(np.float32) for b in BASE], axis=1)


def slog(x):
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def scalar_features(t: pd.DataFrame) -> dict[str, np.ndarray]:
    lm = np.log1p(t.rna_tx_mean.to_numpy())
    rna = np.column_stack(
        [np.log1p(t[c].to_numpy()) for c in ("rna_at", "rna_b9", "rna_b33", "rna_b129", "rna_b513")]
        + [
            lm,
            np.log1p(t.rna_b33.to_numpy()) - lm,
            np.log1p(t.rna_b129.to_numpy()) - lm,
            t.rna_onset_sharp.to_numpy(),
            slog(t.dist_onset),
            np.log1p(t.tx_pos.to_numpy()),
            np.log1p((t.tx_len - t.tx_pos).to_numpy()),
        ]
    )
    pred = np.column_stack(
        [
            np.log1p(1e3 * t.pred_at.to_numpy()),
            np.log1p(1e3 * t.pred_codon.to_numpy()),
            t.pred_rel.to_numpy(),
            t.pred_share.to_numpy(),
            np.log1p(t.pred_rank.to_numpy()),
            np.log1p(1e3 * t.pred_chx_b33.to_numpy()),
        ]
    )
    ckpt = np.column_stack(
        [
            np.log1p(1e3 * t.ckpt_at.to_numpy()),
            t.ckpt_rel.to_numpy(),
            t.ckpt_share.to_numpy(),
            np.log1p(t.ckpt_rank.to_numpy()),
        ]
    )
    annot = np.column_stack(
        [
            t.in_frame_cds.to_numpy(float),
            slog(t.tx_pos - t.cds_start),
            (t.region == "5UTR").to_numpy(float),
            (t.region == "CDS").to_numpy(float),
        ]
    )
    return {"rna_pos": rna, "pred_final": pred, "pred_v2ref": ckpt, "annotation": annot,
            "onehot43": onehot(t.ctx43)}


def matched_negatives(t: pd.DataFrame, pos: pd.DataFrame, rng) -> pd.Index:
    pool = t[(~t.label) & (~t.near_any_call)]
    groups = {k: list(g.index) for k, g in pool.groupby(["codon", "region", "decile"])}
    coarse = {k: list(g.index) for k, g in pool.groupby(["codon", "region"])}
    used: set = set()
    out = []
    for r in pos.itertuples():
        cand = [i for i in groups.get((r.codon, r.region, r.decile), []) if i not in used]
        if len(cand) < NEG_PER_POS:
            cand = [i for i in coarse.get((r.codon, r.region), []) if i not in used]
        take = rng.choice(cand, size=min(NEG_PER_POS, len(cand)), replace=False)
        used.update(take.tolist())
        out.extend(take.tolist())
    return pd.Index(out)


def build_sets(t: pd.DataFrame, rng) -> dict[str, pd.DataFrame]:
    t = t[t.dense_row >= 0]
    nc_pos = t[t.label & t.codon.isin(NC)]
    atg_pos = t[t.label & (t.codon == "ATG")]
    sets = {}
    for name, pos in (("nc_global", nc_pos), ("atg_global", atg_pos)):
        neg = matched_negatives(t, pos, rng)
        sets[name] = pd.concat([pos, t.loc[neg]])
    # local: uncalled cognate codons within +-LOCAL nt of an NC start, same transcript
    keep = []
    site_of = np.full(len(t), -1)
    by_tx = {k: g for k, g in t.groupby("transcript_id")}
    for r in nc_pos.itertuples():
        g = by_tx[r.transcript_id]
        near = g[(~g.label) & ((g.tx_pos - r.tx_pos).abs() <= LOCAL)]
        keep.extend(near.index.tolist())
    local = pd.concat([nc_pos, t.loc[sorted(set(keep))]])
    sets["nc_local"] = local
    del site_of
    # reproducible NC starts: both replicate tracks peak within 1 nt of the start
    rep = nc_pos[(nc_pos.rep1_off.abs() <= 1) & (nc_pos.rep2_off.abs() <= 1)]
    sets["ncrep_global"] = pd.concat([rep, t.loc[matched_negatives(t, rep, rng)]])
    keep = set()
    for r in rep.itertuples():
        g = by_tx[r.transcript_id]
        keep.update(g[(~g.label) & ((g.tx_pos - r.tx_pos).abs() <= LOCAL)].index.tolist())
    sets["ncrep_local"] = pd.concat([rep, t.loc[sorted(keep)]])
    return sets


def fit_eval(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Out-of-fold probability (outer 5-fold grouped, inner 3-fold grouped C choice)."""
    grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0) if x.shape[1] > 64 else (1e-2, 1e-1, 1.0, 10.0)
    oof = np.zeros(len(y))
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for tr, te in outer.split(x, y, groups):
        best, best_c = -1.0, grid[0]
        if len(grid) > 1:
            for c in grid:
                scores = []
                for itr, ite in GroupKFold(3).split(x[tr], y[tr], groups[tr]):
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
                    m.fit(x[tr][itr], y[tr][itr])
                    scores.append(average_precision_score(y[tr][ite],
                                                          m.predict_proba(x[tr][ite])[:, 1]))
                if np.mean(scores) > best:
                    best, best_c = float(np.mean(scores)), c
        m = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=3000))
        m.fit(x[tr], y[tr])
        oof[te] = m.predict_proba(x[te])[:, 1]
    return oof


def local_win(s: pd.DataFrame, score: np.ndarray) -> float:
    """Mean over NC starts of the share of uncalled cognate codons within +-LOCAL it outranks."""
    s = s.assign(score=score)
    wins = []
    by_tx = {k: g for k, g in s.groupby("transcript_id")}
    for r in s[s.label].itertuples():
        g = by_tx[r.transcript_id]
        near = g[(~g.label) & ((g.tx_pos - r.tx_pos).abs() <= LOCAL)]
        if len(near):
            wins.append(float((r.score > near.score).mean() + 0.5 * (r.score == near.score).mean()))
    return float(np.mean(wins)) if wins else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/p6nc"))
    ap.add_argument("--dense", type=Path, default=Path("data/dense_exp_store"))
    ap.add_argument("--out", type=Path, default=Path("data/p6nc/probe.tsv"))
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma list of feature combos to run")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t = pd.read_parquet(args.data / "codons.parquet")
    tx_mean = t.groupby("transcript_id").rna_tx_mean.first()
    dec = pd.qcut(tx_mean.rank(method="first"), 10, labels=False)
    t["decile"] = t.transcript_id.map(dec).astype(int)
    sets = build_sets(t, rng)
    rows_all = np.unique(np.concatenate([s.index.to_numpy() for s in sets.values()]))
    print({k: (len(s), int(s.label.sum())) for k, s in sets.items()}, flush=True)

    # gather every feature for rows_all (row order = rows_all)
    sub = t.loc[rows_all]
    feats = scalar_features(sub)
    v2 = np.load(args.data / "v2emb.npy", mmap_mode="r")
    feats["v2_ft_pca256"] = np.asarray(v2[rows_all, 0], np.float32)
    feats["v2_ft_pca256_codon"] = np.asarray(v2[rows_all], np.float32).reshape(len(rows_all), -1)
    for key in ("emb_base", "emb_final"):
        path = args.data / f"{key}.npy"
        if path.exists():
            feats[key] = np.asarray(np.load(path, mmap_mode="r")[rows_all], np.float32)
    dense_rows = sub.dense_row.to_numpy()
    order = np.argsort(dense_rows)
    for key, rel in DENSE_KEYS.items():
        arr = np.load(args.dense / rel, mmap_mode="r")
        x = np.empty((len(dense_rows), arr.shape[1]), np.float32)
        x[order] = arr[dense_rows[order]]
        feats[key] = x
        print(f"loaded {key} {x.shape}", flush=True)
    pos_of = pd.Series(np.arange(len(rows_all)), index=rows_all)

    combos = {name: [name] for name in feats}
    for emb in ("emb_base", "emb_final", "v2_ft_pca256", "ag16k_store", "evo2_w8k", "evo2_txp"):
        if emb in feats:
            combos[f"{emb}+rna_pos"] = [emb, "rna_pos"]
    combos["onehot43+rna_pos"] = ["onehot43", "rna_pos"]
    combos["pred_final+rna_pos"] = ["pred_final", "rna_pos"]
    if "emb_base" in feats:
        combos["emb_base+rna_pos+pred_final"] = ["emb_base", "rna_pos", "pred_final"]
        combos["emb_base+evo2_w8k+rna_pos"] = ["emb_base", "evo2_w8k", "rna_pos"]

    for extra in ("onehot43", "emb_final", "emb_base", "evo2_w8k", "annotation"):
        if extra in feats:
            combos[f"pred_final+{extra}"] = ["pred_final", extra]
    if args.only:
        combos = {k: v for k, v in combos.items() if k in set(args.only.split(","))}
    jobs = []
    for set_name, s in sets.items():
        idx = pos_of.loc[s.index].to_numpy()
        y = s.label.to_numpy(int)
        groups = s.transcript_id.to_numpy()
        for combo, parts in combos.items():
            jobs.append((set_name, combo, idx, y, groups, parts))

    def run(set_name, combo, idx, y, groups, parts):
        x = np.concatenate([feats[p][idx] for p in parts], axis=1)
        oof = fit_eval(x, y, groups, args.seed)
        return set_name, combo, x.shape[1], oof

    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run)(*j) for j in jobs)
    out = []
    for set_name, combo, dim, oof in results:
        s = sets[set_name]
        y = s.label.to_numpy(int)
        row = {"set": set_name, "features": combo, "dim": dim, "n": len(y), "pos": int(y.sum()),
               "auroc": roc_auc_score(y, oof), "auprc": average_precision_score(y, oof)}
        if set_name.endswith("_local"):
            row["local_win"] = local_win(s, oof)
        out.append(row)
    # raw, unfitted scores of the final and reference models
    for set_name, s in sets.items():
        y = s.label.to_numpy(int)
        for col in ("pred_share", "pred_at", "pred_rel", "ckpt_share", "ckpt_at", "real_share"):
            score = s[col].to_numpy(float)
            row = {"set": set_name, "features": f"raw:{col}", "dim": 1, "n": len(y),
                   "pos": int(y.sum()), "auroc": roc_auc_score(y, score),
                   "auprc": average_precision_score(y, score)}
            if set_name.endswith("_local"):
                row["local_win"] = local_win(s, score)
            out.append(row)
    table = pd.DataFrame(out).sort_values(["set", "auroc"], ascending=[True, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, sep="\t", index=False)
    for set_name, s in sets.items():
        s[["transcript_id", "tx_pos", "codon", "label", "region", "decile"]].to_parquet(
            args.out.parent / f"set_{set_name}.parquet"
        )
    with pd.option_context("display.width", 200, "display.precision", 3, "display.max_rows", 500):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
