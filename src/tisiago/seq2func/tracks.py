"""Stranded 1-bp Ribo-seq tracks: RPF P-sites (Ribo-TISH offsets) and RNA 5' ends."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pyBigWig
import pysam

from tisiago.seq2func.psite import psite, rna_end5
from tisiago.seq2func.samples import load_samples, parse_para

STRANDS = {"+": "plus", "-": "minus"}


def track_prefix(out_dir: str | Path, condition: str, library: str, rep: int | str) -> Path:
    """Path prefix of one track; ``rep='pooled'`` for the replicate sum."""
    return Path(out_dir) / condition / f"{library}_rep{rep}"


def count_positions(
    bam_path: str | Path, position_fn: Callable
) -> dict[tuple[str, str], Counter]:
    """Count reads per (chrom, strand) at the position returned by ``position_fn``."""
    counts: dict[tuple[str, str], Counter] = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            position = position_fn(read)
            if position is None:
                continue
            key = (read.reference_name, "-" if read.is_reverse else "+")
            counts.setdefault(key, Counter())[position] += 1
    return counts


def write_bigwigs(
    counts: dict[tuple[str, str], Counter], chrom_sizes: list[tuple[str, int]], prefix: Path
) -> int:
    """Write ``prefix.plus.bw`` / ``prefix.minus.bw`` at 1-bp resolution; return the total."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    for strand, name in STRANDS.items():
        bw = pyBigWig.open(str(prefix.with_name(f"{prefix.name}.{name}.bw")), "w")
        bw.addHeader(chrom_sizes)
        for chrom, size in chrom_sizes:
            counter = counts.get((chrom, strand))
            if not counter:
                continue
            positions = sorted(p for p in counter if 0 <= p < size)
            if not positions:
                continue
            values = [float(counter[p]) for p in positions]
            bw.addEntries(chrom, positions, values=values, span=1)
            total += int(sum(values))
        bw.close()
    return total


def read_bigwig_counts(path: str | Path) -> dict[str, Counter]:
    """Read a 1-bp bigWig back into per-chromosome position counters."""
    result: dict[str, Counter] = {}
    bw = pyBigWig.open(str(path))
    try:
        for chrom in bw.chroms():
            counter = Counter()
            for start, end, value in bw.intervals(chrom) or ():
                for position in range(start, end):
                    counter[position] += value
            result[chrom] = counter
    finally:
        bw.close()
    return result


def _chrom_sizes(bam_path: str | Path) -> list[tuple[str, int]]:
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return list(zip(bam.references, bam.lengths, strict=True))


def build_sample(sample, out_dir: Path) -> dict:
    """Build one sample's stranded track and write its depth record."""
    if sample.library == "rna":
        position_fn = rna_end5
    else:
        offsets, m0 = parse_para(sample.para)
        position_fn = lambda read: psite(read, offsets, m0)  # noqa: E731
    prefix = track_prefix(out_dir, sample.condition, sample.library, sample.rep)
    counts = count_positions(sample.bam, position_fn)
    total = write_bigwigs(counts, _chrom_sizes(sample.bam), prefix)
    record = {
        "condition": sample.condition,
        "library": sample.library,
        "rep": int(sample.rep),
        "sample_id": sample.sample_id,
        "total": total,
    }
    prefix.with_suffix(".json").write_text(json.dumps(record) + "\n")
    return record


def pool(out_dir: Path, condition: str, library: str, chrom_sizes) -> int:
    """Sum the two replicate tracks into ``{library}_reppooled``."""
    counts: dict[tuple[str, str], Counter] = {}
    for rep in (1, 2):
        prefix = track_prefix(out_dir, condition, library, rep)
        for strand, name in STRANDS.items():
            for chrom, counter in read_bigwig_counts(f"{prefix}.{name}.bw").items():
                counts.setdefault((chrom, strand), Counter()).update(counter)
    return write_bigwigs(counts, chrom_sizes, track_prefix(out_dir, condition, library, "pooled"))


def main() -> None:
    """CLI: build one sample (``--index``, a Slurm array task) or pool all replicates."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atis-root", default=os.environ.get("TISIAGO_ATIS_DATA"))
    ap.add_argument("--out-dir", default="data/p6/tracks")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--pool", action="store_true")
    args = ap.parse_args()
    samples = load_samples(args.atis_root)
    out_dir = Path(args.out_dir)
    if args.index is not None:
        print(json.dumps(build_sample(samples.iloc[args.index], out_dir)))
        return
    sizes = _chrom_sizes(samples.bam.iloc[0])
    for (condition, library), _ in samples.groupby(["condition", "library"]):
        print(condition, library, pool(out_dir, condition, library, sizes), flush=True)


if __name__ == "__main__":
    main()
