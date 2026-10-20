"""TIS (harringtonine) peak share at real Ribo-TISH starts on val: real pooled vs predicted float.

Usage:
    python autoresearch/p6/tis_peak_share.py --pred NAME=DIR [--pred NAME=DIR ...] \
        [--condition K562,HeLa] [--out results.tsv]

DIR is a ``predict_tracks --mode float`` output (``<DIR>/<C>/tis_pooled.{plus,minus}.bw``).
Sites: unique start anchors of the pooled real calls (``$TISIAGO_RIBOTISH_RUN``, Jimmy's filter)
on chr7 inside the val windows. Per site the TIS profile +-30 nt (5'->3'), normalised to 1;
``meta_share`` = mean normalised profile at offset 0 (the start nt), ``argmax0`` = fraction of
sites whose profile peaks at offset 0, ``within1`` at |offset| <= 1. Only sites with real signal
in the window are kept (same sites for every source). Calls are used for evaluation only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from tisiago.seq2func.ribotish import filter_predictions
from tisiago.seq2func.samples import CONDITIONS

FLANK = 30


def sites_of(calls_run: Path, condition: str, windows: pd.DataFrame) -> pd.DataFrame:
    """Unique (chrom, strand, anchor) of the filtered pooled calls inside the val windows."""
    path = calls_run / condition / "pooled" / f"{condition}_pooled.calls.tsv"
    calls = filter_predictions(pd.read_csv(path, sep="\t"))
    pattern = r"^(?P<chrom>[^:]+):(?P<lo>\d+)-(?P<hi>\d+):(?P<strand>[+-])$"
    pos = calls.GenomePos.str.extract(pattern)
    lo, hi = pos.lo.astype(int), pos.hi.astype(int)
    sites = pd.DataFrame(
        {
            "chrom": pos.chrom.values,
            "strand": pos.strand.values,
            "anchor": np.where(pos.strand == "+", lo, hi),
            "codon": calls.StartCodon.values,
            "tistype": calls.TisType.values,
        }
    )
    sites = sites[sites.chrom.isin(set(windows.chrom))]
    inside = np.zeros(len(sites), bool)
    for w in windows.itertuples():
        inside |= (
            (sites.chrom.values == w.chrom)
            & (sites.anchor.values - FLANK >= w.start)
            & (sites.anchor.values + FLANK + 1 <= w.end)
        )
    sites = sites[inside].drop_duplicates(["chrom", "strand", "anchor"])
    return sites.reset_index(drop=True)


def profiles(prefix: str, sites: pd.DataFrame) -> np.ndarray:
    """``(n, 2 * FLANK + 1)`` values around each anchor, oriented 5'->3'."""
    out = np.zeros((len(sites), 2 * FLANK + 1))
    for strand, suffix in (("+", "plus"), ("-", "minus")):
        bw = pyBigWig.open(f"{prefix}.{suffix}.bw")
        try:
            for i in np.flatnonzero(sites.strand.values == strand):
                c, a = sites.chrom.iat[i], int(sites.anchor.iat[i])
                v = np.nan_to_num(np.array(bw.values(c, a - FLANK, a + FLANK + 1), float))
                out[i] = v if strand == "+" else v[::-1]
        finally:
            bw.close()
    return out


def summarize(prof: np.ndarray) -> dict:
    """Meta share at offset 0 and per-site argmax statistics of positive profiles."""
    norm = prof / np.maximum(prof.sum(1, keepdims=True), 1e-12)
    meta = norm.mean(0)
    offs = np.arange(-FLANK, FLANK + 1)
    argmax = offs[prof.argmax(1)]
    return {
        "meta_share": float(meta[FLANK]),
        "meta_peak_offset": int(offs[meta.argmax()]),
        "flank_share": float(np.r_[meta[: FLANK - 3], meta[FLANK + 4 :]].mean()),
        "argmax0": float((argmax == 0).mean()),
        "within1": float((np.abs(argmax) <= 1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pred", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--condition", default="K562")
    ap.add_argument("--calls-run", type=Path, default=os.environ.get("TISIAGO_RIBOTISH_RUN"))
    ap.add_argument("--tracks-dir", type=Path, default=Path("data/p6/tracks"))
    ap.add_argument("--windows", type=Path, default=Path("data/p6/windows_new/val.parquet"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.calls_run is None:
        ap.error("--calls-run is required (or set TISIAGO_RIBOTISH_RUN)")
    windows = pd.read_parquet(args.windows)
    if set(windows.chrom) - {"chr7"}:
        ap.error("val windows only (chr7)")
    preds = dict(p.split("=", 1) for p in args.pred)
    conditions = CONDITIONS if args.condition == "all" else args.condition.split(",")
    rows = []
    for condition in conditions:
        sites = sites_of(args.calls_run, condition, windows)
        real = profiles(str(args.tracks_dir / condition / "tis_reppooled"), sites)
        keep = real.sum(1) > 0
        sources = {"real": real}
        for name, d in preds.items():
            sources[name] = profiles(str(Path(d) / condition / "tis_pooled"), sites)
        groups = {
            "all": np.ones(len(sites), bool),
            "ATG": sites.codon.values == "ATG",
            "near_cognate": sites.codon.values != "ATG",
            "Annotated": sites.tistype.values == "Annotated",
        }
        for group, m in groups.items():
            sel = keep & m
            for name, prof in sources.items():
                ok = sel & (prof.sum(1) > 0)
                rows.append(
                    {
                        "condition": condition,
                        "group": group,
                        "source": name,
                        "sites": int(sel.sum()),
                        **summarize(prof[ok]),
                    }
                )
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(table.to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
