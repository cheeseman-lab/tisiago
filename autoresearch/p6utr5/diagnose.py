"""5'UTR start diagnosis of the step-2 P6 model on val K562 (chr7 only; Q1-Q4 of the brief).

Usage:
    python autoresearch/p6utr5/diagnose.py --out data/p6utr5

Predictions are read back from the val export bigWigs into transcript space (exactly the
values export_bigwigs wrote, on the MANE exons of the cached val transcripts): step 2
(``data/p6/txtrain/export/<run>/K562``), step 1 (``data/p6/fastloop/export_twostage/<run>/K562``)
and the genomic model (``ckpt.npy``). Sites: (a) ``cache`` = sites.parquet (real pooled
Ribo-TISH starts, Jimmy's filter); (b) ``rtish`` = Ribo-TISH calls on the real tracks of the
synthetic pipeline (``calls/K562/real``) that lie on a MANE exon of a K562-expressed val
transcript, mapped to MANE mRNA positions (TisType is Ribo-TISH's for its own transcript).
A source hits a site when its argmax over +-30 nt is within 1 nt (``fastloop.site_hits``).
Writes ``sites.parquet`` (one row per site x set) and ``transcripts.parquet``.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func import fastloop
from tisiago.seq2func.predict_tracks import strand_shift
from tisiago.seq2func.scorecard import BigWigs
from tisiago.seq2func.txcache import TxCache, spliced_values, unit_of

COND = "K562"
FLANK = fastloop.FLANK  # 30
W = 15


def group(t: str) -> str:
    t = str(t)
    for g in ("Annotated", "Extended", "Truncated", "5'UTR", "3'UTR"):
        if t.startswith(g):
            return g
    return "Internal/Novel"


def read_export(path: Path, units) -> dict[str, np.ndarray]:
    reader = BigWigs()
    out = {}
    for tid, unit in units.items():
        f = path / f"tis_pooled.{'plus' if unit.strand == '+' else 'minus'}.bw"
        out[tid] = spliced_values(reader, f, unit, strand_shift(unit.strand))
    reader.close()
    return out


def rtish_sites(calls: Path, units, gtf: Path) -> pd.DataFrame:
    ex, strand = collections.defaultdict(list), {}
    for line in open(gtf):
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if f[2] != "exon":
            continue
        tid = f[8].split('transcript_id "')[1].split('"')[0]
        ex[tid].append((int(f[3]) - 1, int(f[4])))
        strand[tid] = f[6]

    def gpos(tid, i):
        for s, t in sorted(ex[tid], reverse=strand[tid] == "-"):
            if i < t - s:
                return s + i if strand[tid] == "+" else t - 1 - i
            i -= t - s
        return None

    lookup = {}
    for tid, unit in units.items():
        for i, p in enumerate(unit.positions):
            lookup.setdefault((unit.strand, int(p)), (tid, i))
    d = pd.read_csv(calls, sep="\t")
    rows = []
    for r in d.itertuples():
        g = gpos(r.Tid, int(r.Start))
        hit = lookup.get((strand[r.Tid], g)) if g is not None else None
        if hit:
            rows.append(
                {"transcript_id": hit[0], "tx_pos": hit[1], "codon": r.StartCodon, "tistype": r.TisType}
            )
    out = pd.DataFrame(rows)
    # one row per MANE position; prefer a 5'UTR label when several Ribo-TISH transcripts call it
    out["_p"] = ~out.tistype.str.startswith("5'UTR")
    return out.sort_values("_p").drop_duplicates(["transcript_id", "tx_pos"]).drop(columns="_p")


def window_stats(vec: np.ndarray, pos: int, w: int = W) -> dict:
    lo, hi = max(pos - w, 0), min(pos + w + 1, len(vec))
    win = vec[lo:hi]
    tot = vec.sum()
    ws = win.sum()
    s3 = vec[max(pos - 1, 0) : pos + 2].sum()
    am = int(np.argmax(win)) + lo - pos if ws > 0 else 0
    return {
        "win_frac": ws / tot if tot > 0 else np.nan,  # window mass / transcript mass
        "site_frac": vec[pos] / tot if tot > 0 else np.nan,  # height at the start nt
        "share3": s3 / ws if ws > 0 else np.nan,  # +-1 nt share of the +-15 window
        "peakshare": win.max() / ws if ws > 0 else np.nan,  # max share of the window (sharpness)
        "argoff": am,
        "abs3": s3,  # absolute +-1 nt sum (track units)
        "bg": tot / len(vec),  # transcript mean per nt
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/p6/txcache/fc2d_k562"))
    ap.add_argument("--step2", type=Path, default=Path("data/p6/txtrain/export/sb_k562_s0/K562"))
    ap.add_argument(
        "--step1", type=Path, default=Path("data/p6/fastloop/export_twostage/ts_k562_s0/K562")
    )
    ap.add_argument(
        "--calls",
        type=Path,
        default=Path("data/p6/txtrain/export/sb_k562_s0/predicted_val_floatresample_s0/calls/K562"),
    )
    ap.add_argument("--out", type=Path, default=Path("data/p6utr5"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cache = TxCache(args.cache)
    k = cache.condition_index(COND)
    val = cache.rows("val")
    assert set(val.chrom) == {"chr7"}, val.chrom.unique()
    val = val[val[f"expressed_{COND}"].to_numpy(bool)]
    units = {r.transcript_id: unit_of(r) for r in val.itertuples()}
    tis = cache.array("val_chr7", "tis")
    rna = cache.array("val_chr7", "rna")
    seq = cache.array("val_chr7", "seq")
    ckpt = cache.array("val_chr7", "ckpt")
    tx = {}
    for r in val.itertuples():
        lo, hi = int(r.offset), int(r.offset) + int(r.length)
        a = np.asarray(tis[k, lo:hi], np.float64)
        tx[r.transcript_id] = {
            "rep1": a[:, 0],
            "rep2": a[:, 1],
            "real": a[:, 2],
            "genomic": np.asarray(ckpt[k, lo:hi, 0], np.float64),
            "rna": np.asarray(rna[k, lo:hi, 0], np.float64),
            "seq": np.asarray(seq[lo:hi]),
            "row": r,
        }
    for name, path in (("step2", args.step2), ("step1", args.step1)):
        for tid, v in read_export(path, units).items():
            tx[tid][name] = v
    sources = ("real", "rep1", "rep2", "step2", "step1", "genomic")

    sets = {"cache": cache.sites[cache.sites.condition == COND]}
    sets["cache"] = sets["cache"][sets["cache"].transcript_id.isin(units)]
    sets["rtish"] = rtish_sites(
        args.calls / "real" / f"{COND}_TIS_predict.txt", units, args.calls / "annotation.chr7.gtf"
    )
    pred_calls = rtish_sites(
        args.calls / "predicted" / f"{COND}_TIS_predict.txt", units, args.calls / "annotation.chr7.gtf"
    )
    pred_call_keys = set(zip(pred_calls.transcript_id, pred_calls.tx_pos))

    rows = []
    for set_name, sites in sets.items():
        for s in sites.itertuples():
            v = tx[s.transcript_id]
            r = v["row"]
            pos = int(s.tx_pos)
            rec = {
                "set": set_name,
                "transcript_id": s.transcript_id,
                "gene": r.gene_name,
                "tx_pos": pos,
                "codon": s.codon,
                "atg": str(s.codon).upper() == "ATG",
                "tistype": s.tistype,
                "group": group(s.tistype),
                "length": int(r.length),
                "cds_start": int(r.cds_start),
                "dist_cds": int(r.cds_start) - pos,  # >0: upstream of the annotated start
                "first_exon": (pos < int(r.junctions[0])) if len(r.junctions) else True,
                "rtish_pred_call": (s.transcript_id, pos) in pred_call_keys,
            }
            for src in sources:
                prof = fastloop.site_profiles(v[src], [pos])
                rec[f"{src}_hit"] = bool(fastloop.site_hits(prof)[0])
                for key, val_ in window_stats(v[src], pos).items():
                    rec[f"{src}_{key}"] = val_
            rows.append(rec)
    sites = pd.DataFrame(rows)

    trows = []
    for tid, v in tx.items():
        r = v["row"]
        cs, ce = int(r.cds_start), int(r.cds_end)
        rec = {
            "transcript_id": tid,
            "gene": r.gene_name,
            "length": int(r.length),
            "cds_start": cs,
            "utr5_len": cs,
            "rna_per_nt": v["rna"].sum() / len(v["rna"]),
            "real_tis_total": v["real"].sum(),
        }
        cds_rna = v["rna"][cs:ce].mean() if ce > cs else np.nan
        rec["rna_cds_mean"] = cds_rna
        rec["rna_utr5_mean"] = v["rna"][:cs].mean() if cs > 0 else np.nan
        # RNA-supported 5' end: first position whose 25-nt RNA mean reaches 10% of the CDS mean
        if cs > 0 and cds_rna > 0:
            sm = np.convolve(v["rna"][: cs + 25], np.ones(25) / 25, mode="valid")
            above = np.flatnonzero(sm >= 0.1 * cds_rna)
            rec["rna_start"] = int(above[0]) if len(above) else cs
        else:
            rec["rna_start"] = 0
        for src in ("real", "rep1", "rep2", "step2", "step1", "genomic"):
            x = v[src]
            tot = x.sum()
            rec[f"{src}_utr5"] = x[: max(cs - W, 0)].sum() / tot if tot > 0 else np.nan
            rec[f"{src}_ann"] = x[max(cs - W, 0) : cs + W + 1].sum() / tot if tot > 0 else np.nan
        trows.append(rec)
    trans = pd.DataFrame(trows)
    sites = sites.merge(
        trans[["transcript_id", "rna_per_nt", "real_tis_total", "rna_start", "rna_utr5_mean",
               "rna_cds_mean"]],
        on="transcript_id",
    )  # fmt: skip
    for src in ("real", "step2"):
        ann = {}
        for tid, v in tx.items():
            cs = int(v["row"].cds_start)
            x = v[src]
            ann[tid] = x[max(cs - W, 0) : cs + W + 1].sum() / x.sum() if x.sum() > 0 else np.nan
        sites[f"{src}_annwin_frac"] = sites.transcript_id.map(ann)
    sites.to_parquet(args.out / "sites.parquet")
    trans.to_parquet(args.out / "transcripts.parquet")
    print(f"{len(sites)} site rows, {len(trans)} transcripts -> {args.out}")


if __name__ == "__main__":
    main()
