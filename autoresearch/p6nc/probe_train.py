"""Q1, data-rich variant: probes trained on TRAIN chromosomes, scored on the chr7 sets of probe.py.

Usage:
    python autoresearch/p6nc/probe_train.py --data data/p6nc --out data/p6nc/probe_train.tsv

The chr7 probes of ``probe.py`` see ~5k rows for 1.5-4k-dim features, so a weak probe could be
sample-limited. Here the probe is fit on the dense store's train-split rows (chr7/8/9 never used
for fitting): K562 pooled Ribo-TISH starts (reprocessed run, Jimmy's filter) that fall on exons of
K562-expressed MANE train transcripts of the txcache index, vs uncalled codons of the same identity
and region (no filtered call of any condition within +-30 nt genomic, same strand), 5 per start.
C is chosen by 3-fold CV grouped by chromosome. The fitted probe then scores the chr7 rows of
``set_<name>.parquet`` written by probe.py. Labels are used for evaluation only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import DENSE_KEYS, NC, local_win  # noqa: E402

from tisiago.seq2func.ribotish import filter_predictions  # noqa: E402
from tisiago.seq2func.samples import CONDITIONS  # noqa: E402
from tisiago.seq2func.txcache import TxCache, unit_of  # noqa: E402

FORBIDDEN = {"chr7", "chr8", "chr9"}
EXCLUDE = 30
NEG_PER_POS = 5


def call_keys(calls_run: Path, condition: str) -> pd.DataFrame:
    """Filtered pooled calls as dense-store keys (chrom, strand, gstart, codon)."""
    path = calls_run / condition / "pooled" / f"{condition}_pooled.calls.tsv"
    calls = filter_predictions(pd.read_csv(path, sep="\t", usecols=[
        "GenomePos", "StartCodon", "TISPvalue", "RiboPvalue", "FisherQvalue"]))
    pos = calls.GenomePos.str.extract(r"^(?P<chrom>[^:]+):(?P<lo>\d+)-(?P<hi>\d+):(?P<strand>[+-])$")
    gstart = np.where(pos.strand == "+", pos.lo.astype(int), pos.hi.astype(int))
    return pd.DataFrame({"chrom": pos.chrom, "strand": pos.strand, "gstart": gstart,
                         "codon": calls.StartCodon.values}).drop_duplicates(
        ["chrom", "strand", "gstart"])


def exon_lookup(cache: TxCache, condition: str) -> dict:
    """(chrom, strand) -> (sorted first-nt genomic positions, region code 0/1/2) of exonic bases."""
    idx = cache.index
    idx = idx[(idx.split == "train") & idx[f"expressed_{condition}"].to_numpy(bool)]
    parts: dict = {}
    for row in idx.itertuples():
        if row.chrom in FORBIDDEN:
            continue
        u = unit_of(row)
        p = u.positions
        code = np.ones(len(p), np.int8)
        code[: int(row.cds_start)] = 0
        code[int(row.cds_end):] = 2
        parts.setdefault((row.chrom, row.strand), []).append((p, code))
    out = {}
    for key, items in parts.items():
        p = np.concatenate([a for a, _ in items])
        c = np.concatenate([b for _, b in items])
        o = np.argsort(p, kind="stable")
        out[key] = (p[o], c[o])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/p6nc"))
    ap.add_argument("--dense", type=Path, default=Path("data/dense_exp_store"))
    ap.add_argument("--cache", type=Path, default=Path("data/p6/txcache/ar_rnacond_k562_v2"))
    ap.add_argument("--calls-run", type=Path, default=os.environ.get("TISIAGO_RIBOTISH_RUN"))
    ap.add_argument("--condition", default="K562")
    ap.add_argument("--out", type=Path, default=Path("data/p6nc/probe_train.tsv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    m = pd.read_parquet(args.dense / "manifest.parquet",
                        columns=["row_idx", "chrom", "strand", "gstart", "codon", "split"])
    tr = m[(m.split == "train") & ~m.chrom.isin(FORBIDDEN)]
    tr = tr.drop_duplicates(["chrom", "strand", "gstart"]).reset_index(drop=True)
    assert not set(tr.chrom) & FORBIDDEN

    # region / expressed-MANE membership of each train row (first nt genomic position)
    lookup = exon_lookup(TxCache(args.cache), args.condition)
    first = np.where(tr.strand == "+", tr.gstart, tr.gstart - 1)
    region = np.full(len(tr), -1, np.int8)
    for (chrom, strand), (p, c) in lookup.items():
        sel = np.flatnonzero((tr.chrom.values == chrom) & (tr.strand.values == strand))
        i = np.searchsorted(p, first[sel])
        ok = (i < len(p)) & (p[np.minimum(i, len(p) - 1)] == first[sel])
        region[sel[ok]] = c[i[ok]]
    tr["region"] = region
    tr = tr[tr.region >= 0]

    calls = {c: call_keys(args.calls_run, c) for c in CONDITIONS}
    k562 = calls[args.condition]
    key = pd.MultiIndex.from_frame(tr[["chrom", "strand", "gstart"]])
    tr["label"] = key.isin(pd.MultiIndex.from_frame(k562[["chrom", "strand", "gstart"]]))
    near = np.zeros(len(tr), bool)
    allc = pd.concat(calls.values()).drop_duplicates(["chrom", "strand", "gstart"])
    for (chrom, strand), g in allc.groupby(["chrom", "strand"]):
        sel = np.flatnonzero((tr.chrom.values == chrom) & (tr.strand.values == strand))
        if not len(sel):
            continue
        cs = np.sort(g.gstart.to_numpy())
        x = tr.gstart.values[sel]
        i = np.searchsorted(cs, x)
        d = np.minimum(np.abs(x - cs[np.clip(i - 1, 0, len(cs) - 1)]),
                       np.abs(x - cs[np.clip(i, 0, len(cs) - 1)]))
        near[sel] = d <= EXCLUDE
    tr["near"] = near
    print(f"train rows in expressed MANE exons: {len(tr)}; K562 starts {int(tr.label.sum())}",
          flush=True)

    def training_set(codons) -> pd.DataFrame:
        pos = tr[tr.label & tr.codon.isin(codons)]
        pool = tr[~tr.label & ~tr.near & tr.codon.isin(codons)]
        neg = []
        for (codon, reg), g in pos.groupby(["codon", "region"]):
            cand = pool[(pool.codon == codon) & (pool.region == reg)].index.to_numpy()
            neg.extend(rng.choice(cand, size=min(len(cand), NEG_PER_POS * len(g)),
                                  replace=False).tolist())
        return pd.concat([pos, tr.loc[neg]])

    train_sets = {"nc": training_set(NC), "atg": training_set(("ATG",))}
    for k, s in train_sets.items():
        print(f"train set {k}: {len(s)} rows, {int(s.label.sum())} starts", flush=True)

    codons = pd.read_parquet(args.data / "codons.parquet",
                             columns=["transcript_id", "tx_pos", "dense_row"])
    eval_sets = {}
    for name in ("nc_global", "nc_local", "ncrep_global", "ncrep_local", "atg_global"):
        path = args.data / f"set_{name}.parquet"
        if path.exists():
            s = pd.read_parquet(path)
            s["dense_row"] = codons.loc[s.index, "dense_row"].to_numpy()
            eval_sets[name] = s
    arrays = {k: np.load(args.dense / rel, mmap_mode="r") for k, rel in DENSE_KEYS.items()}

    def gather(key, rows):
        rows = np.asarray(rows)
        o = np.argsort(rows)
        x = np.empty((len(rows), arrays[key].shape[1]), np.float32)
        x[o] = arrays[key][rows[o]]
        return x

    combos = {k: [k] for k in DENSE_KEYS}
    combos["ag16k_store+evo2_w8k"] = ["ag16k_store", "evo2_w8k"]
    out = []
    for train_name, s in train_sets.items():
        rows = s.row_idx.to_numpy()
        y = s.label.to_numpy(int)
        groups = s.chrom.to_numpy()
        targets = [n for n in eval_sets if n.startswith(train_name)]
        for combo, parts in combos.items():
            x = np.concatenate([gather(p, rows) for p in parts], axis=1)
            best, best_c = -1.0, None
            for c in (1e-4, 1e-3, 1e-2, 1e-1):
                sc = []
                for a, b in GroupKFold(3).split(x, y, groups):
                    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
                    mdl.fit(x[a], y[a])
                    sc.append(average_precision_score(y[b], mdl.predict_proba(x[b])[:, 1]))
                if np.mean(sc) > best:
                    best, best_c = float(np.mean(sc)), c
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=3000))
            mdl.fit(x, y)
            for name in targets:
                e = eval_sets[name]
                xe = np.concatenate([gather(p, e.dense_row.to_numpy()) for p in parts], axis=1)
                score = mdl.predict_proba(xe)[:, 1]
                ye = e.label.to_numpy(int)
                row = {"train": train_name, "train_rows": len(y), "train_pos": int(y.sum()),
                       "C": best_c, "train_cv_auprc": best, "set": name, "features": combo,
                       "auroc": roc_auc_score(ye, score),
                       "auprc": average_precision_score(ye, score)}
                if name.endswith("_local"):
                    row["local_win"] = local_win(e, score)
                out.append(row)
                print(row, flush=True)
    table = pd.DataFrame(out)
    table.to_csv(args.out, sep="\t", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
