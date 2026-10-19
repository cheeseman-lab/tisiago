"""TIS (harringtonine) peak placement and sharpness at real Ribo-TISH calls: real vs predicted tracks.

Usage: python autoresearch/p6/diag_tis_peaks.py <calls_dir> [<calls_dir> ...]
Each calls_dir is a predicted_calls output (real/, tracks_real/, tracks_predicted/). Predicted
tracks of all given dirs are summed (e.g. two Poisson seeds); real tracks come from the first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from tisiago.seq2func.ribotish import filter_predictions

FLANK = 30


def start_positions(calls: pd.DataFrame) -> pd.DataFrame:
    """Genomic start-codon anchor per call from GenomePos (start end of the ORF span)."""
    pos = calls.GenomePos.str.extract(r"^(?P<chrom>[^:]+):(?P<lo>\d+)-(?P<hi>\d+):(?P<strand>[+-])$")
    lo, hi = pos.lo.astype(int), pos.hi.astype(int)
    return pd.DataFrame(
        {
            "chrom": pos.chrom,
            "strand": pos.strand,
            "anchor": np.where(pos.strand == "+", lo, hi),
            "codon": calls.StartCodon.values,
            "tistype": calls.TisType.values,
        }
    )


def profiles(track_dirs: list[Path], sites: pd.DataFrame) -> np.ndarray:
    """(n_sites, 2*FLANK+1) TIS counts around each anchor, oriented 5'->3', reps and dirs summed."""
    out = np.zeros((len(sites), 2 * FLANK + 1))
    for d in track_dirs:
        for rep in (1, 2):
            for strand, suffix in (("+", "plus"), ("-", "minus")):
                bw = pyBigWig.open(str(d / f"tis_rep{rep}.{suffix}.bw"))
                for i in np.flatnonzero(sites.strand.values == strand):
                    c, a = sites.chrom.iat[i], int(sites.anchor.iat[i])
                    v = np.nan_to_num(np.array(bw.values(c, a - FLANK, a + FLANK + 1), float))
                    out[i] += v if strand == "+" else v[::-1]
                bw.close()
    return out


def summarize(name: str, prof: np.ndarray, mask: np.ndarray) -> dict:
    p = prof[mask]
    keep = p.sum(1) > 0
    p = p[keep]
    norm = p / p.sum(1, keepdims=True)
    meta = norm.mean(0)
    offs = np.arange(-FLANK, FLANK + 1)
    argmax = offs[p.argmax(1)]
    return {
        "arm": name,
        "sites": int(mask.sum()),
        "with_signal": int(keep.sum()),
        "meta_peak_offset": int(offs[meta.argmax()]),
        "meta_peak_share": round(float(meta.max()), 4),
        "per_site_peak_at_meta_peak": round(float((argmax == offs[meta.argmax()]).mean()), 3),
        "per_site_peak_within_1": round(float((np.abs(argmax - offs[meta.argmax()]) <= 1).mean()), 3),
        "median_mass_in_peak_pm1": round(
            float(np.median(norm[:, FLANK + offs[meta.argmax()] - 1 : FLANK + offs[meta.argmax()] + 2].sum(1))),
            3,
        ),
    }


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    calls = filter_predictions(pd.read_csv(dirs[0] / "real" / "K562_TIS_predict.txt", sep="\t"))
    sites = start_positions(calls).drop_duplicates(["chrom", "strand", "anchor"]).reset_index(drop=True)
    real = profiles([dirs[0] / "tracks_real" / "K562"], sites)
    pred = profiles([d / "tracks_predicted" / "K562" for d in dirs], sites)
    groups = {
        "all": np.ones(len(sites), bool),
        "ATG": sites.codon.values == "ATG",
        "near_cognate": sites.codon.values != "ATG",
        "Annotated": sites.tistype.values == "Annotated",
    }
    rows = []
    for g, m in groups.items():
        for name, prof in (("real", real), ("predicted", pred)):
            rows.append({"group": g, **summarize(name, prof, m)})
    print(pd.DataFrame(rows).to_string(index=False))
    offs = np.arange(-FLANK, FLANK + 1)
    for name, prof in (("real", real), ("predicted", pred)):
        p = prof[prof.sum(1) > 0]
        meta = (p / p.sum(1, keepdims=True)).mean(0)
        print(name, "meta -6..+6:", " ".join(f"{o:+d}:{m:.3f}" for o, m in zip(offs, meta) if -6 <= o <= 6))


if __name__ == "__main__":
    main()
