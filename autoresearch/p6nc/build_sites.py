"""Per-codon table for the near-cognate diagnosis (val chr7, K562, MANE transcripts of the txcache).

Usage:
    python autoresearch/p6nc/build_sites.py --out data/p6nc

Every cognate codon (ATG + 9 near-cognates) of every K562-expressed val transcript of
``data/p6/txcache/ar_rnacond_k562_v2`` becomes one row, in mRNA coordinates (5'->3'). Labels are
K562 pooled Ribo-TISH starts (reprocessed run, Jimmy's filter, ``sites.parquet``). Per row:
region, codon context, dense-store row (genomic join, for base AlphaGenome / Evo2 features),
RNA/position features, the final model's predicted TIS (``ar_rnacond_k562_fc2d_l2w300``,
K562 held out) and the reference model's own prediction (``ckpt.npy``) at the codon.

Writes ``<out>/codons.parquet``, ``<out>/v2emb.npy`` (rows x 3 x 256: fine-tuned v2 PCA embedding
at the codon's three nt) and ``<out>/profiles.npz`` (per labelled site: +-90 nt real rep1, rep2,
pooled, predicted and v2 TIS profiles, plus per-transcript predicted peaks for the error analysis).
Never reads chr8/chr9.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

from tisiago.seq2func.predict_tracks import strand_shift
from tisiago.seq2func.scorecard import BigWigs
from tisiago.seq2func.txcache import TxCache, spliced_values, unit_of

CODONS = ("ATG", "CTG", "GTG", "TTG", "ACG", "AGG", "AAG", "ATT", "ATC", "ATA")
COMP = str.maketrans("ACGTN", "TGCAN")
FLANK = 30
WIDE = 90
EXCLUDE = 30
FORBIDDEN = {"chr8", "chr9"}


def box(x: np.ndarray, w: int) -> np.ndarray:
    """Centred box mean of width ``w`` (zero padded)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    half = w // 2
    idx = np.arange(len(x))
    lo, hi = np.clip(idx - half, 0, len(x)), np.clip(idx + half + 1, 0, len(x))
    return (c[hi] - c[lo]) / w


def local_stats(v: np.ndarray, p: int, flank: int = FLANK) -> tuple[float, float, int]:
    """Value share at ``p`` in +-flank, rank of ``p`` (0 = top), offset of the local argmax."""
    lo, hi = max(p - flank, 0), min(p + flank + 1, len(v))
    win = v[lo:hi]
    s = win.sum()
    share = float(v[p] / s) if s > 0 else 0.0
    rank = int((win > v[p]).sum())
    off = int(lo + win.argmax() - p) if s > 0 else 999
    return share, rank, off


def pad_window(v: np.ndarray, p: int, flank: int) -> np.ndarray:
    padded = np.concatenate([np.zeros(flank), v, np.zeros(flank)])
    return padded[p : p + 2 * flank + 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/p6/txcache/ar_rnacond_k562_v2"))
    ap.add_argument(
        "--pred",
        type=Path,
        default=Path("data/p6/runs/ar_rnacond_k562_fc2d_l2w300/predicted_val_float/K562"),
    )
    ap.add_argument("--dense", type=Path, default=Path("data/dense_exp_store"))
    ap.add_argument("--genome", required=True)
    ap.add_argument("--condition", default="K562")
    ap.add_argument("--out", type=Path, default=Path("data/p6nc"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cache = TxCache(args.cache)
    k = cache.condition_index(args.condition)
    val = cache.rows("val")
    assert set(val.chrom) == {"chr7"} and not FORBIDDEN & set(val.chrom)
    val = val[val[f"expressed_{args.condition}"].to_numpy(bool)].reset_index(drop=True)
    if args.limit:
        val = val.head(args.limit)
    sites = cache.sites
    k_sites = sites[sites.condition == args.condition]
    any_sites = sites.groupby("transcript_id").tx_pos.apply(lambda s: np.unique(s.to_numpy()))
    k_by_tx = {t: g for t, g in k_sites.groupby("transcript_id")}

    dense = pd.read_parquet(args.dense / "manifest.parquet", columns=["row_idx", "chrom", "strand",
                                                                       "gstart", "codon"])
    dense = dense[dense.chrom == "chr7"]
    dense_key = {
        (s, int(g)): (int(r), c)
        for s, g, r, c in zip(dense.strand, dense.gstart, dense.row_idx, dense.codon, strict=True)
    }

    fasta = Fasta(args.genome, as_raw=True, sequence_always_upper=True)
    reader = BigWigs()
    rows, embs, prof_rows = [], [], []
    peak_tables = []
    for t, row in enumerate(val.itertuples()):
        unit = unit_of(row)
        pos = unit.positions
        raw = fasta[unit.chrom][int(pos.min()) : int(pos.max()) + 1]
        seq = "".join(raw[p - int(pos.min())] for p in pos)
        if unit.strand == "-":
            seq = seq.translate(COMP)
        L = len(seq)
        lo, hi = int(row.offset), int(row.offset) + L
        rna = np.asarray(cache.array(row.shard, "rna")[k, lo:hi, 0], np.float64)
        tis = np.asarray(cache.array(row.shard, "tis")[k, lo:hi], np.float64)
        ckpt = np.asarray(cache.array(row.shard, "ckpt")[k, lo:hi, 0], np.float64)
        emb = cache.array(row.shard, "emb")
        name = "plus" if unit.strand == "+" else "minus"
        shift = strand_shift(unit.strand)
        pred = spliced_values(reader, args.pred / f"tis_pooled.{name}.bw", unit, shift)
        pred_chx = spliced_values(reader, args.pred / f"chx_pooled.{name}.bw", unit, shift)
        assert len(pred) == L

        rna_mean = rna.mean()
        b = {w: box(rna, w) for w in (9, 33, 129, 513)}
        b33_med = np.median(b[33][b[33] > 0]) if (b[33] > 0).any() else 0.0
        onset_idx = np.flatnonzero(b[33] >= 0.2 * b33_med) if b33_med > 0 else np.array([0])
        onset = int(onset_idx[0]) if len(onset_idx) else 0
        chx33 = box(pred_chx, 33)
        pred_max = pred.max() if pred.max() > 0 else 1.0
        ckpt_max = ckpt.max() if ckpt.max() > 0 else 1.0
        # predicted peaks: argmax of own +-15 window and >= 5% of the transcript maximum
        pk = [
            p
            for p in range(L)
            if pred[p] >= 0.05 * pred_max and pred[p] == pred[max(p - 15, 0) : p + 16].max()
        ]
        peak_tables.append(pd.DataFrame({"transcript_id": row.transcript_id, "tx_pos": pk,
                                         "pred": pred[pk], "pred_rel": pred[pk] / pred_max}))

        calls_here = any_sites.get(row.transcript_id, np.array([], int))
        kpos = k_by_tx.get(row.transcript_id)
        kmap = {int(p): (c, ty) for p, c, ty in zip(kpos.tx_pos, kpos.codon, kpos.tistype,
                                                     strict=True)} if kpos is not None else {}
        cds_s, cds_e = int(row.cds_start), int(row.cds_end)
        for p in range(L - 2):
            codon = seq[p : p + 3]
            if codon not in CODONS and p not in kmap:
                continue
            region = "5UTR" if p < cds_s else ("CDS" if p < cds_e else "3UTR")
            near_call = bool(len(calls_here) and np.abs(calls_here - p).min() <= EXCLUDE)
            g = int(pos[p]) + (0 if unit.strand == "+" else 1)
            drow, dcodon = dense_key.get((unit.strand, g), (-1, None))
            ps, pr, po = local_stats(pred, p)
            cs, cr, co = local_stats(ckpt, p)
            rs, rr, ro = local_stats(tis[:, 2], p)
            r1s, r1r, r1o = local_stats(tis[:, 0], p)
            r2s, r2r, r2o = local_stats(tis[:, 1], p)
            ctx = seq[max(p - 20, 0) : p + 23]
            ctx = "N" * (20 - p) + ctx if p < 20 else ctx
            ctx = ctx + "N" * (43 - len(ctx))
            label = p in kmap
            rows.append(
                {
                    "transcript_id": row.transcript_id,
                    "gene_name": row.gene_name,
                    "strand": unit.strand,
                    "tx_pos": p,
                    "tx_len": L,
                    "codon": codon,
                    "label": label,
                    "call_codon": kmap[p][0] if label else None,
                    "tistype": kmap[p][1] if label else None,
                    "region": region,
                    "cds_start": cds_s,
                    "cds_end": cds_e,
                    "in_frame_cds": (p - cds_s) % 3 == 0,
                    "near_any_call": near_call,
                    "dense_row": drow,
                    "dense_codon": dcodon,
                    "ctx43": ctx,
                    "rna_tx_mean": rna_mean,
                    "rna_at": rna[p],
                    "rna_b9": b[9][p],
                    "rna_b33": b[33][p],
                    "rna_b129": b[129][p],
                    "rna_b513": b[513][p],
                    "rna_onset_sharp": np.log1p(rna[p]) - np.log1p(b[9][p]),
                    "rna_onset": onset,
                    "dist_onset": p - onset,
                    "pred_at": pred[p],
                    "pred_codon": pred[p : p + 3].sum(),
                    "pred_rel": pred[p] / pred_max,
                    "pred_share": ps,
                    "pred_rank": pr,
                    "pred_off": po,
                    "pred_chx_b33": chx33[p],
                    "ckpt_at": ckpt[p],
                    "ckpt_rel": ckpt[p] / ckpt_max,
                    "ckpt_share": cs,
                    "ckpt_rank": cr,
                    "ckpt_off": co,
                    "real_share": rs,
                    "real_rank": rr,
                    "real_off": ro,
                    "real_sum30": tis[max(p - FLANK, 0) : p + FLANK + 1, 2].sum(),
                    "rep1_off": r1o,
                    "rep1_sum30": tis[max(p - FLANK, 0) : p + FLANK + 1, 0].sum(),
                    "rep2_off": r2o,
                    "rep2_sum30": tis[max(p - FLANK, 0) : p + FLANK + 1, 1].sum(),
                }
            )
            idx = np.clip(np.arange(p, p + 3), 0, L - 1) + lo
            embs.append(np.asarray(emb[idx], np.float16))
            if label:
                prof_rows.append(
                    np.stack(
                        [pad_window(v, p, WIDE) for v in (tis[:, 0], tis[:, 1], tis[:, 2], pred,
                                                          ckpt)]
                    )
                )
        if t % 100 == 0:
            print(f"{t}/{len(val)} transcripts, {len(rows)} codons", flush=True)

    table = pd.DataFrame(rows)
    table.to_parquet(args.out / "codons.parquet", index=False)
    np.save(args.out / "v2emb.npy", np.stack(embs))
    pd.concat(peak_tables).to_parquet(args.out / "pred_peaks.parquet", index=False)
    np.savez(args.out / "profiles.npz", profiles=np.stack(prof_rows),
             site_rows=np.flatnonzero(table.label.to_numpy()))
    ok = table.dense_row >= 0
    print(f"codons {len(table)}; labelled {int(table.label.sum())}; dense-joined {ok.mean():.3f}")
    bad = ok & (table.dense_codon != table.codon)
    print(f"dense codon mismatches: {int(bad.sum())}")


if __name__ == "__main__":
    main()
