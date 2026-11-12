"""Track QC: replicate concordance (the accuracy ceiling) and CHX 3-nt periodicity."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tisiago.tracks.psite import RIBOTISH_MINUS_SHIFT
from tisiago.tracks.samples import CONDITIONS, LIBRARIES
from tisiago.tracks.track_io import read_bigwig_counts, track_prefix


def replicate_concordance(
    rep1: dict[str, Counter], rep2: dict[str, Counter], regions: list[tuple[str, int, int]]
) -> float:
    """Pearson correlation of log1p region totals between two replicate tracks."""

    def totals(track):
        out = np.zeros(len(regions), dtype=np.float64)
        by_chrom: dict[str, list[int]] = defaultdict(list)
        for i, (chrom, _, _) in enumerate(regions):
            by_chrom[chrom].append(i)
        for chrom, idx in by_chrom.items():
            counter = track.get(chrom)
            if not counter:
                continue
            pos = np.fromiter(counter.keys(), dtype=np.int64, count=len(counter))
            val = np.fromiter(counter.values(), dtype=np.float64, count=len(counter))
            order = np.argsort(pos)
            pos, cum = pos[order], np.concatenate([[0.0], np.cumsum(val[order])])
            starts = np.array([regions[i][1] for i in idx])
            ends = np.array([regions[i][2] for i in idx])
            out[idx] = (
                cum[np.searchsorted(pos, ends, "left")] - cum[np.searchsorted(pos, starts, "left")]
            )
        return out

    a, b = np.log1p(totals(rep1)), np.log1p(totals(rep2))
    return float(np.corrcoef(a, b)[0, 1])


def _frame_counts(
    counts: dict[str, Counter],
    cds: list[tuple[str, str, list[tuple[int, int]]]],
    shift: int = RIBOTISH_MINUS_SHIFT,
) -> tuple[int, int]:
    """Return (in-frame, total) CDS P-site counts; minus-strand positions read at ``p + shift``."""
    in_frame = total = 0
    for chrom, strand, exons in cds:
        track = counts.get(chrom, {})
        positions = [p for start, end in sorted(exons) for p in range(start, end)]
        if strand == "-":
            positions = [p + shift for p in positions[::-1]]
        for index, position in enumerate(positions):
            value = track.get(position, 0)
            total += value
            if index % 3 == 0:
                in_frame += value
    return in_frame, total


def frame_fraction(
    counts: dict[str, Counter], cds: list[tuple[str, str, list[tuple[int, int]]]]
) -> float:
    """Fraction of CDS P-site counts on the first base of a codon (same-strand track)."""
    in_frame, total = _frame_counts(counts, cds)
    return in_frame / total if total else float("nan")


def stranded_frame_fraction(
    plus: dict[str, Counter],
    minus: dict[str, Counter],
    cds: list[tuple[str, str, list[tuple[int, int]]]],
) -> float:
    """Frame fraction using the .plus track for + transcripts and .minus for - transcripts."""
    in_p, tot_p = _frame_counts(plus, [c for c in cds if c[1] == "+"])
    in_m, tot_m = _frame_counts(minus, [c for c in cds if c[1] == "-"])
    total = tot_p + tot_m
    return (in_p + in_m) / total if total else float("nan")


def load_mane(gtf: str | Path):
    """Return (transcript regions [(chrom, start, end)], cds [(chrom, strand, intervals)])."""
    span: dict[str, list] = {}
    cds: dict[str, list] = {}
    pattern = re.compile(r'transcript_id "([^"]+)"')
    with open(gtf) as handle:
        for line in handle:
            if line.startswith("#") or 'tag "MANE_Select"' not in line:
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("exon", "CDS"):
                continue
            tid = pattern.search(f[8]).group(1)
            start, end = int(f[3]) - 1, int(f[4])
            if f[2] == "exon":
                s = span.setdefault(tid, [f[0], start, end])
                s[1], s[2] = min(s[1], start), max(s[2], end)
            else:
                cds.setdefault(tid, [f[0], f[6], []])[2].append((start, end))
    return [tuple(v) for v in span.values()], [tuple(v) for v in cds.values()]


def main() -> None:
    """Write data/tracks/qc.tsv: replicate concordance and CHX frame fraction per track set."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks-dir", default="data/tracks/tracks")
    ap.add_argument("--gtf", required=True, help="GENCODE v49 GTF (MANE_Select tagged)")
    ap.add_argument("--out", default="data/tracks/qc.tsv")
    args = ap.parse_args()
    regions, cds = load_mane(args.gtf)

    def both(condition, library, rep):
        prefix = str(track_prefix(args.tracks_dir, condition, library, rep))
        return read_bigwig_counts(f"{prefix}.plus.bw"), read_bigwig_counts(f"{prefix}.minus.bw")

    def merged(plus, minus):
        out = {c: Counter(v) for c, v in plus.items()}
        for chrom, counter in minus.items():
            out.setdefault(chrom, Counter()).update(counter)
        return out

    rows = []
    for condition in CONDITIONS:
        for library in LIBRARIES:
            r1, r2 = merged(*both(condition, library, 1)), merged(*both(condition, library, 2))
            row = {
                "condition": condition,
                "library": library,
                "replicate_r": replicate_concordance(r1, r2, regions),
                "frame_fraction": float("nan"),
            }
            if library == "chx":
                plus, minus = both(condition, library, "pooled")
                row["frame_fraction"] = stranded_frame_fraction(plus, minus, cds)
            rows.append(row)
            print(row, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
