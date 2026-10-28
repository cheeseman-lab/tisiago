"""Per-site table for the later-exon near-cognate diagnosis (val chr7 only, CPU).

For every Ribo-TISH pooled start of ``--condition`` on expressed val MANE transcripts (and a
control set of near-cognate codons in the same transcripts that are not starts) writes one row
with: replicate / model hits (argmax within 1 nt over +-30 nt, as ``fastloop.rep_summary``),
exon region relative to MANE, isoform 5'-end distances from the GENCODE chr7 annotation,
RNA 5'-end features upstream / downstream of the site (the line's own RNA, ``rna.npy`` sense),
expression, junction distance, codon, frame / position relative to the MANE CDS, TisType.

Model predictions are the exported val float bigWigs of the given step-2 runs
(``autoresearch/p6sharp/sharpness.py:export_predictions``).

Usage:
    python autoresearch/p6nclater/sites.py --cache data/p6/txcache/fc2d_k562 --condition K562 \
        --runs sbflux_k562_s0 sbflux_k562_s1 sbflux_k562_s2 \
        --gtf <calls dir>/annotation.chr7.gtf --out data/p6nclater
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.seq2func import fastloop
from tisiago.seq2func.txcache import TxCache, unit_of

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p6sharp"))
from sharpness import export_predictions  # noqa: E402

NC_CODONS = ("CTG", "GTG", "TTG", "ACG", "ATC", "ATT", "ATA", "AGG", "AAG")
BASES = np.array(list("ACGTN"))
RNA_WIN = 200
CONTROL_EXCLUDE = 30
CONTROLS_PER_TX = 40


def load_gtf(path: Path) -> pd.DataFrame:
    """Exons of every transcript with 5'->3' rank and mRNA offset, plus transcript tags."""
    tx_attr = {}
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("exon", "transcript"):
                continue
            tid = re.search(r'transcript_id "([^"]+)"', f[8]).group(1)
            if f[2] == "transcript":
                tags = set(re.findall(r'tag "([^"]+)"', f[8]))
                ttype = re.search(r'transcript_type "([^"]+)"', f[8]).group(1)
                tx_attr[tid] = (
                    ttype,
                    "mRNA_start_NF" in tags,
                    "basic" in tags,
                    "MANE_Select" in tags,
                )
            else:
                rows.append((f[0], f[6], tid, int(f[3]) - 1, int(f[4])))
    ex = pd.DataFrame(rows, columns=["chrom", "strand", "tid", "start", "end"])
    ex["sign"] = np.where(ex.strand == "+", 1, -1)
    ex["key"] = ex.start * ex.sign
    ex = ex.sort_values(["tid", "key"]).reset_index(drop=True)
    ex["length"] = ex.end - ex.start
    ex["rank"] = ex.groupby("tid").cumcount()
    ex["offset"] = ex.groupby("tid").length.cumsum() - ex.length
    attr = pd.DataFrame.from_dict(
        tx_attr, orient="index", columns=["ttype", "start_nf", "basic", "mane"]
    )
    ex = ex.join(attr, on="tid")
    return ex.drop(columns=["sign", "key"])


class IsoformIndex:
    """Exons containing a genomic position (same strand) and the isoform 5' distance there."""

    def __init__(self, exons: pd.DataFrame):  # noqa: D107
        self.by_strand = {}
        for strand, ex in exons.groupby("strand"):
            ex = ex.sort_values("start").reset_index(drop=True)
            self.by_strand[strand] = (ex, ex.start.to_numpy(), int(ex.length.max()))

    def query(self, strand: str, g: int) -> dict:
        """Min isoform 5' distance (all / full-length / basic / protein-coding) at ``g``."""
        ex, starts, maxlen = self.by_strand[strand]
        lo = np.searchsorted(starts, g - maxlen, "left")
        hi = np.searchsorted(starts, g, "right")
        cand = ex.iloc[lo:hi]
        cand = cand[cand.end > g]
        nan = float("nan")
        if cand.empty:
            return {
                "iso_n": 0,
                "d_all": nan,
                "d_full": nan,
                "d_basic": nan,
                "d_pc": nan,
                "first_full": False,
                "first_basic": False,
                "first_all": False,
            }
        d = cand.offset + np.where(strand == "+", g - cand.start, cand.end - 1 - g)
        full = ~cand.start_nf.to_numpy(bool)
        basic = full & cand.basic.to_numpy(bool)
        pc = full & (cand.ttype == "protein_coding").to_numpy()
        first = (cand["rank"] == 0).to_numpy()
        d = d.to_numpy(float)

        def dmin(mask):
            return float(d[mask].min()) if mask.any() else nan

        return {
            "iso_n": int(cand.tid.nunique()),
            "d_all": dmin(np.ones(len(d), bool)),
            "d_full": dmin(full),
            "d_basic": dmin(basic),
            "d_pc": dmin(pc),
            "first_full": bool((first & full).any()),
            "first_basic": bool((first & basic).any()),
            "first_all": bool(first.any()),
        }


def window_sum(v: np.ndarray, lo: int, hi: int) -> float:
    lo, hi = max(lo, 0), min(hi, len(v))
    return float(v[lo:hi].sum()) if hi > lo else 0.0


def rna_features(rna: np.ndarray, p: int, tx_density: float) -> dict:
    """RNA 5'-end mass up/downstream of ``p`` on the MANE mRNA (sense, depth-scaled)."""
    up = window_sum(rna, p - RNA_WIN, p)
    down = window_sum(rna, p, p + RNA_WIN)
    up_n = min(RNA_WIN, p)
    down_n = min(RNA_WIN, len(rna) - p)
    eps = 0.05
    up_d = up / max(up_n, 1)
    down_d = down / max(down_n, 1)
    seg = rna[max(p - RNA_WIN, 0) : p]
    if len(seg) >= 5:
        smooth = np.convolve(seg, np.ones(5) / 5, mode="valid")
        peak = float(smooth.max())
    else:
        peak = float("nan")
    return {
        "rna_up": up_d,
        "rna_down": down_d,
        "rna_step": float(np.log2((down_d + eps) / (up_d + eps))) if up_n >= 50 else np.nan,
        "rna_peak_rel": peak / tx_density if tx_density > 0 else float("nan"),
        "rna_local_rel": (up + down) / max(up_n + down_n, 1) / tx_density
        if tx_density > 0
        else float("nan"),
    }


def frame_class(p: int, cds_start: int, cds_end: int) -> str:
    if p < cds_start:
        return "upstream_inframe" if (cds_start - p) % 3 == 0 else "upstream_offframe"
    if p < cds_end:
        return "cds_inframe" if (p - cds_start) % 3 == 0 else "cds_offframe"
    return "downstream"


def build(args) -> pd.DataFrame:
    cache = TxCache(args.cache)
    if cache.config.get("split") == "test":
        raise ValueError("validation only")
    c = args.condition
    k = cache.condition_index(c)
    val = cache.rows("val")
    assert set(val.chrom) == {"chr7"}, "validation is chr7 only"
    units = val[val[f"expressed_{c}"].to_numpy(bool)]
    preds = {run: export_predictions(cache, args.export / run, c) for run in args.runs}
    iso = IsoformIndex(load_gtf(args.gtf))
    sites = cache.sites[
        (cache.sites.condition == c) & cache.sites.transcript_id.isin(units.transcript_id)
    ]
    by_tx = dict(tuple(sites.groupby("transcript_id")))
    rng = np.random.default_rng(0)
    out = []
    for row in units.itertuples():
        lo, hi = int(row.offset), int(row.offset) + int(row.length)
        tis = np.rint(np.asarray(cache.array(row.shard, "tis")[k, lo:hi], np.float64))
        r1, r2 = tis[:, 0], tis[:, 1]
        chx = np.asarray(cache.array(row.shard, "chx")[k, lo:hi, 2], np.float64)
        rna = np.asarray(cache.array(row.shard, "rna")[k, lo:hi, 0], np.float64)
        seq = np.asarray(cache.array(row.shard, "seq")[lo:hi])
        codons = np.array(
            ["".join(BASES[seq[i : i + 3]]) for i in range(len(seq) - 2)] + ["NNN", "NNN"]
        )
        model = [
            np.asarray(preds[run][row.transcript_id][k, :, 0], np.float64) for run in args.runs
        ]
        positions = unit_of(row).positions
        junctions = np.asarray(row.junctions, int)
        first_j = int(junctions[0]) if len(junctions) else None
        tx_density = float(rna.sum() / len(rna))
        group = by_tx.get(row.transcript_id)
        real_pos = group.tx_pos.to_numpy(int) if group is not None else np.zeros(0, int)
        entries = []
        if group is not None:
            for s in group.itertuples():
                entries.append((int(s.tx_pos), str(s.codon), str(s.tistype), True))
        # controls: near-cognate codons that are not within +-30 nt of a real start
        nc_pos = np.flatnonzero(np.isin(codons, NC_CODONS))
        if len(real_pos):
            far = np.min(np.abs(nc_pos[:, None] - real_pos[None, :]), axis=1) > CONTROL_EXCLUDE
            nc_pos = nc_pos[far]
        if len(nc_pos) > CONTROLS_PER_TX:
            nc_pos = np.sort(rng.choice(nc_pos, CONTROLS_PER_TX, replace=False))
        entries += [(int(p), codons[p], "control", False) for p in nc_pos]
        if not entries:
            continue
        pos = np.array([e[0] for e in entries])
        prof1 = fastloop.site_profiles(r1, pos)
        prof2 = fastloop.site_profiles(r2, pos)
        hit1, hit2 = fastloop.site_hits(prof1), fastloop.site_hits(prof2)
        profc = fastloop.site_profiles(chx, pos)
        mhits, mrel, moff = [], [], []
        for m in model:
            pm = fastloop.site_profiles(m, pos)
            mhits.append(fastloop.site_hits(pm))
            moff.append(pm.argmax(1) - fastloop.FLANK)
            mx = pm.max(1)
            mrel.append(np.where(mx > 0, pm[:, fastloop.FLANK] / np.maximum(mx, 1e-12), np.nan))
        mhits, mrel, moff = np.mean(mhits, 0), np.nanmean(mrel, 0), moff[0]
        tis_tx, chx_tx = float((r1 + r2).sum()), float(chx.sum())
        regions = fastloop.exon_regions(pos, first_j)
        for i, (p, codon, tistype, real) in enumerate(entries):
            up_j = junctions[junctions <= p]
            cs = int(row.cds_start)
            n_atg_up = int(np.sum(codons[: max(p, 0)] == "ATG"))
            n_atg_inframe = (
                int(np.sum(codons[cs:p:3] == "ATG")) if p > cs and (p - cs) % 3 == 0 else -1
            )
            pw = (prof1[i] + prof2[i])[fastloop.FLANK - 1 : fastloop.FLANK + 2].sum()
            cw = profc[i][fastloop.FLANK - 1 : fastloop.FLANK + 2].sum()
            g = int(positions[p])
            rec = {
                "condition": c,
                "transcript_id": row.transcript_id,
                "gene_name": row.gene_name,
                "tx_pos": p,
                "genomic": g,
                "strand": row.strand,
                "real": real,
                "codon": codon,
                "cls": "atg" if codon == "ATG" else "nc",
                "tistype": tistype,
                "region": regions[i],
                "n_junctions_up": int(len(up_j)),
                "dist_junction_up": int(p - up_j.max()) if len(up_j) else -1,
                "tx_len": int(row.length),
                "cds_start": int(row.cds_start),
                "frame": frame_class(p, int(row.cds_start), int(row.cds_end)),
                "dist_cds_start": int(p - row.cds_start),
                "tx_rna_density": tx_density,
                "tx_tis_reads": float(r1.sum() + r2.sum()),
                "rep1_hit": bool(hit1[i]),
                "rep2_hit": bool(hit2[i]),
                "rep1_win": float(prof1[i].sum()),
                "rep2_win": float(prof2[i].sum()),
                "model_hit": float(mhits[i]),
                "model_rel": float(mrel[i]),
                "model_off_s0": int(moff[i]),
                "n_atg_up": n_atg_up,
                "n_atg_inframe_between": n_atg_inframe,
                "tis_site": float(pw),
                "chx_site": float(cw),
                "chx_win": float(profc[i].sum()),
                "tis_chx_enrich": float(
                    np.log2(((pw + 0.5) / max(tis_tx, 1)) / ((cw + 0.5) / max(chx_tx, 1)))
                ),
            }
            rec.update(rna_features(rna, p, tx_density))
            rec.update(iso.query(row.strand, g))
            out.append(rec)
    table = pd.DataFrame(out)
    # MANE 5' distance from the annotation must equal tx_pos (sanity on coordinates)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--export", type=Path, default=Path("data/p6/txtrain/export"))
    ap.add_argument("--out", type=Path, default=Path("data/p6nclater"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table = build(args)
    path = args.out / f"sites_{args.condition.lower()}.parquet"
    table.to_parquet(path, index=False)
    real = table[table.real]
    print(
        f"{args.condition}: {len(real)} starts, {len(table) - len(real)} controls; "
        f"d_all<=tx_pos {(real.d_all <= real.tx_pos).mean():.3f} -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
